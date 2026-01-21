// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "arrow/array/data.h"
#include "arrow/type_traits.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/logging.h"
#include "arrow/util/logging_internal.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace ree_util {

// Macro listing all value types supported by run-end encoding.
#define ARROW_REE_SUPPORTED_TYPES(ACTION)               \
  ACTION(Boolean, BOOL)                                 \
  ACTION(Int8, INT8)                                    \
  ACTION(Int16, INT16)                                  \
  ACTION(Int32, INT32)                                  \
  ACTION(Int64, INT64)                                  \
  ACTION(UInt8, UINT8)                                  \
  ACTION(UInt16, UINT16)                                \
  ACTION(UInt32, UINT32)                                \
  ACTION(UInt64, UINT64)                                \
  ACTION(Float, FLOAT)                                  \
  ACTION(Double, DOUBLE)                                \
  ACTION(HalfFloat, HALF_FLOAT)                         \
  ACTION(Date32, DATE32)                                \
  ACTION(Date64, DATE64)                                \
  ACTION(Time32, TIME32)                                \
  ACTION(Time64, TIME64)                                \
  ACTION(Timestamp, TIMESTAMP)                          \
  ACTION(Duration, DURATION)                            \
  ACTION(MonthInterval, INTERVAL_MONTHS)                \
  ACTION(DayTimeInterval, INTERVAL_DAY_TIME)            \
  ACTION(MonthDayNanoInterval, INTERVAL_MONTH_DAY_NANO) \
  ACTION(Decimal32, DECIMAL32)                          \
  ACTION(Decimal64, DECIMAL64)                          \
  ACTION(Decimal128, DECIMAL128)                        \
  ACTION(Decimal256, DECIMAL256)                        \
  ACTION(FixedSizeBinary, FIXED_SIZE_BINARY)            \
  ACTION(String, STRING)                                \
  ACTION(Binary, BINARY)                                \
  ACTION(LargeString, LARGE_STRING)                     \
  ACTION(LargeBinary, LARGE_BINARY)

/// \brief Get the child array holding the run ends from an REE array
inline const ArraySpan& RunEndsArray(const ArraySpan& span) { return span.child_data[0]; }

/// \brief Get the child array holding the data values from an REE array
inline const ArraySpan& ValuesArray(const ArraySpan& span) { return span.child_data[1]; }

/// \brief Get a pointer to run ends values of an REE array
template <typename RunEndCType>
const RunEndCType* RunEnds(const ArraySpan& span) {
  assert(RunEndsArray(span).type->id() == CTypeTraits<RunEndCType>::ArrowType::type_id);
  return RunEndsArray(span).GetValues<RunEndCType>(1);
}

/// \brief Perform basic validations on the parameters of an REE array
/// and its two children arrays
///
/// All the checks complete in O(1) time. Consequently, this function:
/// - DOES NOT check that run_ends is sorted and all-positive
/// - DOES NOT check the actual contents of the run_ends and values arrays
Status ValidateRunEndEncodedChildren(const RunEndEncodedType& type,
                                     int64_t logical_length,
                                     const std::shared_ptr<ArrayData>& run_ends_data,
                                     const std::shared_ptr<ArrayData>& values_data,
                                     int64_t null_count, int64_t logical_offset);

/// \brief Compute the logical null count of an REE array
int64_t LogicalNullCount(const ArraySpan& span);

namespace internal {

/// \brief Uses binary-search to find the physical offset given a logical offset
/// and run-end values
///
/// \return the physical offset or run_ends_size if the physical offset is not
/// found in run_ends
template <typename RunEndCType>
int64_t FindPhysicalIndex(const RunEndCType* run_ends, int64_t run_ends_size, int64_t i,
                          int64_t absolute_offset) {
  assert(absolute_offset + i >= 0);
  auto it = std::upper_bound(run_ends, run_ends + run_ends_size, absolute_offset + i);
  int64_t result = std::distance(run_ends, it);
  assert(result <= run_ends_size);
  return result;
}

/// \brief Uses binary-search to calculate the range of physical values (and
/// run-ends) necessary to represent the logical range of values from
/// offset to length
///
/// \return a pair of physical offset and physical length
template <typename RunEndCType>
std::pair<int64_t, int64_t> FindPhysicalRange(const RunEndCType* run_ends,
                                              int64_t run_ends_size, int64_t length,
                                              int64_t offset) {
  const int64_t physical_offset =
      FindPhysicalIndex<RunEndCType>(run_ends, run_ends_size, 0, offset);
  // The physical length is calculated by finding the offset of the last element
  // and adding 1 to it, so first we ensure there is at least one element.
  if (length == 0) {
    return {physical_offset, 0};
  }
  const int64_t physical_index_of_last = FindPhysicalIndex<RunEndCType>(
      run_ends + physical_offset, run_ends_size - physical_offset, length - 1, offset);

  assert(physical_index_of_last < run_ends_size - physical_offset);
  return {physical_offset, physical_index_of_last + 1};
}

/// \brief Uses binary-search to calculate the number of physical values (and
/// run-ends) necessary to represent the logical range of values from
/// offset to length
template <typename RunEndCType>
int64_t FindPhysicalLength(const RunEndCType* run_ends, int64_t run_ends_size,
                           int64_t length, int64_t offset) {
  auto [_, physical_length] =
      FindPhysicalRange<RunEndCType>(run_ends, run_ends_size, length, offset);
  // GH-37107: This is a workaround for GCC 7. GCC 7 doesn't ignore
  // variables in structured binding automatically from unused
  // variables when one of these variables are used.
  // See also: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767
  ARROW_UNUSED(_);
  return physical_length;
}

/// \brief Find the physical index into the values array of the REE ArraySpan
///
/// This function uses binary-search, so it has a O(log N) cost.
template <typename RunEndCType>
int64_t FindPhysicalIndex(const ArraySpan& span, int64_t i, int64_t absolute_offset) {
  const int64_t run_ends_size = RunEndsArray(span).length;
  return FindPhysicalIndex(RunEnds<RunEndCType>(span), run_ends_size, i, absolute_offset);
}

/// \brief Find the physical length of an REE ArraySpan
///
/// The physical length of an REE is the number of physical values (and
/// run-ends) necessary to represent the logical range of values from
/// offset to length.
///
/// Avoid calling this function if the physical length can be established in
/// some other way (e.g. when iterating over the runs sequentially until the
/// end). This function uses binary-search, so it has a O(log N) cost.
template <typename RunEndCType>
int64_t FindPhysicalLength(const ArraySpan& span) {
  return FindPhysicalLength(
      /*run_ends=*/RunEnds<RunEndCType>(span),
      /*run_ends_size=*/RunEndsArray(span).length,
      /*length=*/span.length,
      /*offset=*/span.offset);
}

template <typename RunEndCType>
struct PhysicalIndexFinder;

// non-inline implementations for each run-end type
ARROW_EXPORT int64_t FindPhysicalIndexImpl16(PhysicalIndexFinder<int16_t>& self,
                                             int64_t i);
ARROW_EXPORT int64_t FindPhysicalIndexImpl32(PhysicalIndexFinder<int32_t>& self,
                                             int64_t i);
ARROW_EXPORT int64_t FindPhysicalIndexImpl64(PhysicalIndexFinder<int64_t>& self,
                                             int64_t i);

/// \brief Stateful version of FindPhysicalIndex() that caches the result of
/// the previous search and uses it to optimize the next search.
///
/// When new queries for the physical index of a logical index come in,
/// binary search is performed again but the first candidate checked is the
/// result of the previous search (cached physical index) instead of the
/// midpoint of the run-ends array.
///
/// If that test fails, internal::FindPhysicalIndex() is called with one of the
/// partitions defined by the cached index. If the queried logical indices
/// follow an increasing or decreasing pattern, this first test is much more
/// effective in (1) finding the answer right away (close logical indices belong
/// to the same runs) or (2) discarding many more candidates than probing
/// the midpoint would.
///
/// The most adversarial case (i.e. alternating between 0 and length-1 queries)
/// only adds one extra binary search probe when compared to always starting
/// binary search from the midpoint without any of these optimizations.
///
/// \tparam RunEndCType The numeric type of the run-ends array.
template <typename RunEndCType>
struct PhysicalIndexFinder {
  const ArraySpan array_span;
  const RunEndCType* run_ends;
  int64_t last_physical_index = 0;

  explicit PhysicalIndexFinder(const ArrayData& data)
      : array_span(data),
        run_ends(RunEndsArray(array_span).template GetValues<RunEndCType>(1)) {
    assert(CTypeTraits<RunEndCType>::ArrowType::type_id ==
           ::arrow::internal::checked_cast<const RunEndEncodedType&>(*data.type)
               .run_end_type()
               ->id());
  }

  /// \brief Find the physical index into the values array of the REE array.
  ///
  /// \pre 0 <= i < array_span.length()
  /// \param i the logical index into the REE array
  /// \return the physical index into the values array
  int64_t FindPhysicalIndex(int64_t i) {
    if constexpr (std::is_same_v<RunEndCType, int16_t>) {
      return FindPhysicalIndexImpl16(*this, i);
    } else if constexpr (std::is_same_v<RunEndCType, int32_t>) {
      return FindPhysicalIndexImpl32(*this, i);
    } else {
      static_assert(std::is_same_v<RunEndCType, int64_t>, "Unsupported RunEndCType.");
      return FindPhysicalIndexImpl64(*this, i);
    }
  }
};

}  // namespace internal

/// \brief Find the physical index into the values array of the REE ArraySpan
///
/// This function uses binary-search, so it has a O(log N) cost.
ARROW_EXPORT int64_t FindPhysicalIndex(const ArraySpan& span, int64_t i,
                                       int64_t absolute_offset);

/// \brief Find the physical length of an REE ArraySpan
///
/// The physical length of an REE is the number of physical values (and
/// run-ends) necessary to represent the logical range of values from
/// offset to length.
///
/// Avoid calling this function if the physical length can be established in
/// some other way (e.g. when iterating over the runs sequentially until the
/// end). This function uses binary-search, so it has a O(log N) cost.
ARROW_EXPORT int64_t FindPhysicalLength(const ArraySpan& span);

/// \brief Find the physical range of physical values referenced by the REE in
/// the logical range from offset to offset + length
///
/// \return a pair of physical offset and physical length
ARROW_EXPORT std::pair<int64_t, int64_t> FindPhysicalRange(const ArraySpan& span,
                                                           int64_t offset,
                                                           int64_t length);

/// \brief Run-end encode an array
///
/// This function encodes an array into run-end encoded format. It uses an optimized
/// two-pass algorithm: first counting runs, then preallocating buffers and writing.
///
/// \param[in] input The array to encode as ArrayData
/// \param[in] run_end_type The integer type for run ends (int16, int32, or int64)
/// \param[in] pool Memory pool for allocations
/// \return A RunEndEncodedArrayData
ARROW_EXPORT
Result<std::shared_ptr<ArrayData>> RunEndEncodeArray(
    const std::shared_ptr<ArrayData>& input,
    const std::shared_ptr<DataType>& run_end_type, MemoryPool* pool);

// Publish PhysicalIndexFinder outside of the internal namespace.
template <typename RunEndCType>
using PhysicalIndexFinder = internal::PhysicalIndexFinder<RunEndCType>;

template <typename RunEndCType>
class RunEndEncodedArraySpan {
 private:
  struct PrivateTag {};

 public:
  /// \brief Iterator representing the current run during iteration over a
  /// run-end encoded array
  class Iterator {
   public:
    Iterator(PrivateTag, const RunEndEncodedArraySpan& span, int64_t logical_pos,
             int64_t physical_pos)
        : span(span), logical_pos_(logical_pos), physical_pos_(physical_pos) {}

    /// \brief Return the physical index of the run
    ///
    /// The values array can be addressed with this index to get the value
    /// that makes up the run.
    ///
    /// NOTE: if this Iterator is equal to RunEndEncodedArraySpan::end(),
    /// the value returned is undefined.
    int64_t index_into_array() const { return physical_pos_; }

    /// \brief Return the initial logical position of the run
    ///
    /// If this Iterator is equal to RunEndEncodedArraySpan::end(), this is
    /// the same as RunEndEncodedArraySpan::length().
    int64_t logical_position() const { return logical_pos_; }

    /// \brief Return the logical position immediately after the run.
    ///
    /// Pre-condition: *this != RunEndEncodedArraySpan::end()
    int64_t run_end() const { return span.run_end(physical_pos_); }

    /// \brief Returns the logical length of the run.
    ///
    /// Pre-condition: *this != RunEndEncodedArraySpan::end()
    int64_t run_length() const { return run_end() - logical_pos_; }

    /// \brief Check if the iterator is at the end of the array.
    ///
    /// This can be used to avoid paying the cost of a call to
    /// RunEndEncodedArraySpan::end().
    ///
    /// \return true if the iterator is at the end of the array
    bool is_end(const RunEndEncodedArraySpan& span) const {
      return logical_pos_ >= span.length();
    }

    Iterator& operator++() {
      logical_pos_ = span.run_end(physical_pos_);
      physical_pos_ += 1;
      return *this;
    }

    Iterator operator++(int) {
      const Iterator prev = *this;
      ++(*this);
      return prev;
    }

    Iterator& operator--() {
      physical_pos_ -= 1;
      logical_pos_ = (physical_pos_ > 0) ? span.run_end(physical_pos_ - 1) : 0;
      return *this;
    }

    Iterator operator--(int) {
      const Iterator prev = *this;
      --(*this);
      return prev;
    }

    bool operator==(const Iterator& other) const {
      return logical_pos_ == other.logical_pos_;
    }

    bool operator!=(const Iterator& other) const {
      return logical_pos_ != other.logical_pos_;
    }

   public:
    const RunEndEncodedArraySpan& span;

   private:
    int64_t logical_pos_;
    int64_t physical_pos_;
  };

  // Prevent implicit ArrayData -> ArraySpan conversion in
  // RunEndEncodedArraySpan instantiation.
  explicit RunEndEncodedArraySpan(const ArrayData& data) = delete;

  /// \brief Construct a RunEndEncodedArraySpan from an ArraySpan and new
  /// absolute offset and length.
  ///
  /// RunEndEncodedArraySpan{span, off, len} is equivalent to:
  ///
  ///   span.SetSlice(off, len);
  ///   RunEndEncodedArraySpan{span}
  ///
  /// ArraySpan::SetSlice() updates the null_count to kUnknownNullCount, but
  /// we don't need that here as REE arrays have null_count set to 0 by
  /// convention.
  explicit RunEndEncodedArraySpan(const ArraySpan& array_span, int64_t offset,
                                  int64_t length)
      : array_span_{array_span},
        run_ends_(RunEnds<RunEndCType>(array_span_)),
        length_(length),
        offset_(offset) {
    assert(array_span_.type->id() == Type::RUN_END_ENCODED);
  }

  explicit RunEndEncodedArraySpan(const ArraySpan& array_span)
      : RunEndEncodedArraySpan(array_span, array_span.offset, array_span.length) {}

  int64_t offset() const { return offset_; }
  int64_t length() const { return length_; }

  int64_t PhysicalIndex(int64_t logical_pos) const {
    return internal::FindPhysicalIndex(run_ends_, RunEndsArray(array_span_).length,
                                       logical_pos, offset_);
  }

  /// \brief Create an iterator from a logical position and its
  /// pre-computed physical offset into the run ends array
  ///
  /// \param logical_pos is an index in the [0, length()] range
  /// \param physical_offset the pre-calculated PhysicalIndex(logical_pos)
  Iterator iterator(int64_t logical_pos, int64_t physical_offset) const {
    return Iterator{PrivateTag{}, *this, logical_pos, physical_offset};
  }

  /// \brief Create an iterator from a logical position
  ///
  /// \param logical_pos is an index in the [0, length()] range
  Iterator iterator(int64_t logical_pos) const {
    if (logical_pos < length()) {
      return iterator(logical_pos, PhysicalIndex(logical_pos));
    }
    // If logical_pos is above the valid range, use length() as the logical
    // position and calculate the physical address right after the last valid
    // physical position. Which is the physical index of the last logical
    // position, plus 1.
    return (length() == 0) ? iterator(0, PhysicalIndex(0))
                           : iterator(length(), PhysicalIndex(length() - 1) + 1);
  }

  /// \brief Create an iterator representing the logical begin of the run-end
  /// encoded array
  Iterator begin() const { return iterator(0, PhysicalIndex(0)); }

  /// \brief Create an iterator representing the first invalid logical position
  /// of the run-end encoded array
  ///
  /// \warning Avoid calling end() in a loop, as it will recompute the physical
  /// length of the array on each call (O(log N) cost per call).
  ///
  /// You can write your loops like this instead:
  ///
  /// \code
  /// for (auto it = array.begin(), end = array.end(); it != end; ++it) {
  ///   // ...
  /// }
  /// \endcode
  ///
  /// Or this version that does not look like idiomatic C++, but removes
  /// the need for calling end() completely:
  ///
  /// \code
  /// for (auto it = array.begin(); !it.is_end(array); ++it) {
  ///   // ...
  /// }
  /// \endcode
  Iterator end() const {
    return iterator(length(),
                    (length() == 0) ? PhysicalIndex(0) : PhysicalIndex(length() - 1) + 1);
  }

  // Pre-condition: physical_pos < RunEndsArray(array_span_).length);
  inline int64_t run_end(int64_t physical_pos) const {
    assert(physical_pos < RunEndsArray(array_span_).length);
    // Logical index of the end of the run at physical_pos with offset applied
    const int64_t logical_run_end =
        std::max<int64_t>(static_cast<int64_t>(run_ends_[physical_pos]) - offset(), 0);
    // The current run may go further than the logical length, cap it
    return std::min(logical_run_end, length());
  }

 private:
  const ArraySpan& array_span_;
  const RunEndCType* run_ends_;
  const int64_t length_;
  const int64_t offset_;
};

/// \brief Iterate over two run-end encoded arrays in runs or sub-runs that are
/// inside run boundaries on both inputs
///
/// Both RunEndEncodedArraySpan should have the same logical length. Instances
/// of this iterator only hold references to the RunEndEncodedArraySpan inputs.
template <typename Left, typename Right>
class MergedRunsIterator {
 private:
  using LeftIterator = typename Left::Iterator;
  using RightIterator = typename Right::Iterator;

  MergedRunsIterator(LeftIterator left_it, RightIterator right_it,
                     int64_t common_logical_length, int64_t common_logical_pos)
      : ree_iterators_{std::move(left_it), std::move(right_it)},
        logical_length_(common_logical_length),
        logical_pos_(common_logical_pos) {}

 public:
  /// \brief Construct a MergedRunsIterator positioned at logical position 0.
  ///
  /// Pre-condition: left.length() == right.length()
  MergedRunsIterator(const Left& left, const Right& right)
      : MergedRunsIterator(left.begin(), right.begin(), left.length(), 0) {
    assert(left.length() == right.length());
  }

  static Result<MergedRunsIterator> MakeBegin(const Left& left, const Right& right) {
    if (left.length() != right.length()) {
      return Status::Invalid(
          "MergedRunsIterator expects RunEndEncodedArraySpans of the same length");
    }
    return MergedRunsIterator(left, right);
  }

  static Result<MergedRunsIterator> MakeEnd(const Left& left, const Right& right) {
    if (left.length() != right.length()) {
      return Status::Invalid(
          "MergedRunsIterator expects RunEndEncodedArraySpans of the same length");
    }
    return MergedRunsIterator(left.end(), right.end(), left.length(), left.length());
  }

  /// \brief Return the left RunEndEncodedArraySpan child
  const Left& left() const { return std::get<0>(ree_iterators_).span; }

  /// \brief Return the right RunEndEncodedArraySpan child
  const Right& right() const { return std::get<1>(ree_iterators_).span; }

  /// \brief Return the initial logical position of the run
  ///
  /// If is_end(), this is the same as length().
  int64_t logical_position() const { return logical_pos_; }

  /// \brief Whether the iterator is at logical position 0.
  bool is_begin() const { return logical_pos_ == 0; }

  /// \brief Whether the iterator has reached the end of both arrays
  bool is_end() const { return logical_pos_ == logical_length_; }

  /// \brief Return the logical position immediately after the run.
  ///
  /// Pre-condition: !is_end()
  int64_t run_end() const {
    const auto& left_it = std::get<0>(ree_iterators_);
    const auto& right_it = std::get<1>(ree_iterators_);
    return std::min(left_it.run_end(), right_it.run_end());
  }

  /// \brief returns the logical length of the current run
  ///
  /// Pre-condition: !is_end()
  int64_t run_length() const { return run_end() - logical_pos_; }

  /// \brief Return a physical index into the values array of a given input,
  /// pointing to the value of the current run
  template <size_t input_id>
  int64_t index_into_array() const {
    return std::get<input_id>(ree_iterators_).index_into_array();
  }

  int64_t index_into_left_array() const { return index_into_array<0>(); }
  int64_t index_into_right_array() const { return index_into_array<1>(); }

  MergedRunsIterator& operator++() {
    auto& left_it = std::get<0>(ree_iterators_);
    auto& right_it = std::get<1>(ree_iterators_);

    const int64_t left_run_end = left_it.run_end();
    const int64_t right_run_end = right_it.run_end();

    if (left_run_end < right_run_end) {
      logical_pos_ = left_run_end;
      ++left_it;
    } else if (left_run_end > right_run_end) {
      logical_pos_ = right_run_end;
      ++right_it;
    } else {
      logical_pos_ = left_run_end;
      ++left_it;
      ++right_it;
    }
    return *this;
  }

  MergedRunsIterator operator++(int) {
    MergedRunsIterator prev = *this;
    ++(*this);
    return prev;
  }

  MergedRunsIterator& operator--() {
    auto& left_it = std::get<0>(ree_iterators_);
    auto& right_it = std::get<1>(ree_iterators_);

    // The logical position of each iterator is the run_end() of the previous run.
    const int64_t left_logical_pos = left_it.logical_position();
    const int64_t right_logical_pos = right_it.logical_position();

    if (left_logical_pos < right_logical_pos) {
      --right_it;
      logical_pos_ = std::max(left_logical_pos, right_it.logical_position());
    } else if (left_logical_pos > right_logical_pos) {
      --left_it;
      logical_pos_ = std::max(left_it.logical_position(), right_logical_pos);
    } else {
      --left_it;
      --right_it;
      logical_pos_ = std::max(left_it.logical_position(), right_it.logical_position());
    }
    return *this;
  }

  MergedRunsIterator operator--(int) {
    MergedRunsIterator prev = *this;
    --(*this);
    return prev;
  }

  bool operator==(const MergedRunsIterator& other) const {
    return logical_pos_ == other.logical_position();
  }

  bool operator!=(const MergedRunsIterator& other) const { return !(*this == other); }

 private:
  std::tuple<LeftIterator, RightIterator> ree_iterators_;
  const int64_t logical_length_;
  int64_t logical_pos_;
};

namespace internal {

template <typename ArrowType, bool in_has_validity_buffer,
          bool out_has_validity_buffer = in_has_validity_buffer, typename Enable = void>
struct ReadWriteValue {};

// Numeric and primitive C-compatible types
template <typename ArrowType, bool in_has_validity_buffer, bool out_has_validity_buffer>
class ReadWriteValue<ArrowType, in_has_validity_buffer, out_has_validity_buffer,
                     enable_if_has_c_type<ArrowType>> {
 public:
  using ValueRepr = typename ArrowType::c_type;

 private:
  const uint8_t* input_validity_;
  const uint8_t* input_values_;

  // Needed only by the writing functions
  uint8_t* output_validity_;
  uint8_t* output_values_;

 public:
  explicit ReadWriteValue(const ArraySpan& input_values_array,
                          ArrayData* output_values_array_data)
      : input_validity_(in_has_validity_buffer ? input_values_array.buffers[0].data
                                               : NULLPTR),
        input_values_(input_values_array.buffers[1].data),
        output_validity_((out_has_validity_buffer && output_values_array_data)
                             ? output_values_array_data->buffers[0]->mutable_data()
                             : NULLPTR),
        output_values_(output_values_array_data
                           ? output_values_array_data->buffers[1]->mutable_data()
                           : NULLPTR) {}

  [[nodiscard]] bool ReadValue(ValueRepr* out, int64_t read_offset) const {
    bool valid = true;
    if constexpr (in_has_validity_buffer) {
      valid = bit_util::GetBit(input_validity_, read_offset);
    }
    if constexpr (std::is_same_v<ArrowType, BooleanType>) {
      *out = bit_util::GetBit(input_values_, read_offset);
    } else {
      *out = (reinterpret_cast<const ValueRepr*>(input_values_))[read_offset];
    }
    return valid;
  }

  bool CompareValuesAt(int64_t i, int64_t j) const {
    if constexpr (std::is_same_v<ArrowType, BooleanType>) {
      return bit_util::GetBit(input_values_, i) == bit_util::GetBit(input_values_, j);
    } else {
      return (reinterpret_cast<const ValueRepr*>(input_values_))[i] ==
             (reinterpret_cast<const ValueRepr*>(input_values_))[j];
    }
  }

  void ZeroValidityPadding(int64_t length) const {
    ARROW_DCHECK(output_values_);
    if constexpr (out_has_validity_buffer) {
      ARROW_DCHECK(output_validity_);
      const int64_t validity_buffer_size = bit_util::BytesForBits(length);
      output_validity_[validity_buffer_size - 1] = 0;
    }
  }

  void WriteValue(int64_t write_offset, bool valid, ValueRepr value) const {
    if constexpr (out_has_validity_buffer) {
      bit_util::SetBitTo(output_validity_, write_offset, valid);
    }
    if (valid) {
      if constexpr (std::is_same_v<ArrowType, BooleanType>) {
        bit_util::SetBitTo(output_values_, write_offset, value);
      } else {
        (reinterpret_cast<ValueRepr*>(output_values_))[write_offset] = value;
      }
    }
  }

  void WriteRun(int64_t write_offset, int64_t run_length, bool valid,
                ValueRepr value) const {
    if constexpr (out_has_validity_buffer) {
      bit_util::SetBitsTo(output_validity_, write_offset, run_length, valid);
    }
    if (valid) {
      if constexpr (std::is_same_v<ArrowType, BooleanType>) {
        bit_util::SetBitsTo(reinterpret_cast<uint8_t*>(output_values_), write_offset,
                            run_length, value);
      } else {
        auto* output_values_c = reinterpret_cast<ValueRepr*>(output_values_);
        std::fill(output_values_c + write_offset,
                  output_values_c + write_offset + run_length, value);
      }
    }
  }

  bool Compare(ValueRepr lhs, ValueRepr rhs) const { return lhs == rhs; }
};

// FixedSizeBinary, Decimal128
template <typename ArrowType, bool in_has_validity_buffer, bool out_has_validity_buffer>
class ReadWriteValue<ArrowType, in_has_validity_buffer, out_has_validity_buffer,
                     enable_if_fixed_size_binary<ArrowType>> {
 public:
  using ValueRepr = const uint8_t*;

 private:
  const uint8_t* input_validity_;
  const uint8_t* input_values_;

  // Needed only by the writing functions
  uint8_t* output_validity_;
  uint8_t* output_values_;

  const size_t byte_width_;

 public:
  ReadWriteValue(const ArraySpan& input_values_array, ArrayData* output_values_array_data)
      : input_validity_(in_has_validity_buffer ? input_values_array.buffers[0].data
                                               : NULLPTR),
        input_values_(input_values_array.buffers[1].data),
        output_validity_((out_has_validity_buffer && output_values_array_data)
                             ? output_values_array_data->buffers[0]->mutable_data()
                             : NULLPTR),
        output_values_(output_values_array_data
                           ? output_values_array_data->buffers[1]->mutable_data()
                           : NULLPTR),
        byte_width_(input_values_array.type->byte_width()) {}

  [[nodiscard]] bool ReadValue(ValueRepr* out, int64_t read_offset) const {
    bool valid = true;
    if constexpr (in_has_validity_buffer) {
      valid = bit_util::GetBit(input_validity_, read_offset);
    }
    *out = input_values_ + (read_offset * byte_width_);
    return valid;
  }

  bool CompareValuesAt(int64_t i, int64_t j) const {
    return 0 == memcmp(input_values_ + (i * byte_width_),
                       input_values_ + (j * byte_width_), byte_width_);
  }

  void ZeroValidityPadding(int64_t length) const {
    ARROW_DCHECK(output_values_);
    if constexpr (out_has_validity_buffer) {
      ARROW_DCHECK(output_validity_);
      const int64_t validity_buffer_size = bit_util::BytesForBits(length);
      output_validity_[validity_buffer_size - 1] = 0;
    }
  }

  void WriteValue(int64_t write_offset, bool valid, ValueRepr value) const {
    if constexpr (out_has_validity_buffer) {
      bit_util::SetBitTo(output_validity_, write_offset, valid);
    }
    if (valid) {
      memcpy(output_values_ + (write_offset * byte_width_), value, byte_width_);
    }
  }

  void WriteRun(int64_t write_offset, int64_t run_length, bool valid,
                ValueRepr value) const {
    if constexpr (out_has_validity_buffer) {
      bit_util::SetBitsTo(output_validity_, write_offset, run_length, valid);
    }
    if (valid) {
      uint8_t* ptr = output_values_ + (write_offset * byte_width_);
      for (int64_t i = 0; i < run_length; ++i) {
        memcpy(ptr, value, byte_width_);
        ptr += byte_width_;
      }
    }
  }

  bool Compare(ValueRepr lhs, ValueRepr rhs) const {
    return memcmp(lhs, rhs, byte_width_) == 0;
  }
};

// Binary, String...
template <typename ArrowType, bool in_has_validity_buffer, bool out_has_validity_buffer>
class ReadWriteValue<ArrowType, in_has_validity_buffer, out_has_validity_buffer,
                     enable_if_base_binary<ArrowType>> {
 public:
  using ValueRepr = std::string_view;
  using offset_type = typename ArrowType::offset_type;

 private:
  const uint8_t* input_validity_;
  const offset_type* input_offsets_;
  const uint8_t* input_values_;

  // Needed only by the writing functions
  uint8_t* output_validity_;
  offset_type* output_offsets_;
  uint8_t* output_values_;

 public:
  ReadWriteValue(const ArraySpan& input_values_array, ArrayData* output_values_array_data)
      : input_validity_(in_has_validity_buffer ? input_values_array.buffers[0].data
                                               : NULLPTR),
        input_offsets_(input_values_array.template GetValues<offset_type>(1, 0)),
        input_values_(input_values_array.buffers[2].data),
        output_validity_((out_has_validity_buffer && output_values_array_data)
                             ? output_values_array_data->buffers[0]->mutable_data()
                             : NULLPTR),
        output_offsets_(
            output_values_array_data
                ? output_values_array_data->template GetMutableValues<offset_type>(1, 0)
                : NULLPTR),
        output_values_(output_values_array_data
                           ? output_values_array_data->buffers[2]->mutable_data()
                           : NULLPTR) {}

  [[nodiscard]] bool ReadValue(ValueRepr* out, int64_t read_offset) const {
    bool valid = true;
    if constexpr (in_has_validity_buffer) {
      valid = bit_util::GetBit(input_validity_, read_offset);
    }
    if (valid) {
      const offset_type offset0 = input_offsets_[read_offset];
      const offset_type offset1 = input_offsets_[read_offset + 1];
      *out = std::string_view(reinterpret_cast<const char*>(input_values_ + offset0),
                              offset1 - offset0);
    }
    return valid;
  }

  bool CompareValuesAt(int64_t i, int64_t j) const {
    const offset_type len_i = input_offsets_[i + 1] - input_offsets_[i];
    const offset_type len_j = input_offsets_[j + 1] - input_offsets_[j];
    return len_i == len_j &&
           memcmp(input_values_ + input_offsets_[i], input_values_ + input_offsets_[j],
                  static_cast<size_t>(len_i)) == 0;
  }

  void ZeroValidityPadding(int64_t length) const {
    ARROW_DCHECK(output_values_);
    if constexpr (out_has_validity_buffer) {
      ARROW_DCHECK(output_validity_);
      const int64_t validity_buffer_size = bit_util::BytesForBits(length);
      output_validity_[validity_buffer_size - 1] = 0;
    }
  }

  void WriteValue(int64_t write_offset, bool valid, ValueRepr value) const {
    if constexpr (out_has_validity_buffer) {
      bit_util::SetBitTo(output_validity_, write_offset, valid);
    }
    const offset_type offset0 = output_offsets_[write_offset];
    const offset_type offset1 =
        offset0 + (valid ? static_cast<offset_type>(value.size()) : 0);
    output_offsets_[write_offset + 1] = offset1;
    if (valid) {
      memcpy(output_values_ + offset0, value.data(), value.size());
    }
  }

  void WriteRun(int64_t write_offset, int64_t run_length, bool valid,
                ValueRepr value) const {
    if constexpr (out_has_validity_buffer) {
      bit_util::SetBitsTo(output_validity_, write_offset, run_length, valid);
    }
    if (valid) {
      int64_t i = write_offset;
      offset_type offset = output_offsets_[i];
      while (i < write_offset + run_length) {
        memcpy(output_values_ + offset, value.data(), value.size());
        offset += static_cast<offset_type>(value.size());
        i += 1;
        output_offsets_[i] = offset;
      }
    } else {
      offset_type offset = output_offsets_[write_offset];
      offset_type* begin = output_offsets_ + write_offset + 1;
      std::fill(begin, begin + run_length, offset);
    }
  }

  bool Compare(ValueRepr lhs, ValueRepr rhs) const { return lhs == rhs; }
};

template <typename RunEndType, typename ValueType, bool has_validity_buffer>
class RunEndEncodingLoop {
 public:
  using RunEndCType = typename RunEndType::c_type;

 private:
  using ReadWriteValueType = ReadWriteValue<ValueType, has_validity_buffer>;
  using ValueRepr = typename ReadWriteValueType::ValueRepr;

 private:
  const int64_t input_length_;
  const int64_t input_offset_;
  ReadWriteValueType read_write_value_;
  // Needed only by WriteEncodedRuns()
  RunEndCType* output_run_ends_;

 public:
  explicit RunEndEncodingLoop(const ArraySpan& input_array,
                              ArrayData* output_values_array_data,
                              RunEndCType* output_run_ends)
      : input_length_(input_array.length),
        input_offset_(input_array.offset),
        read_write_value_(input_array, output_values_array_data),
        output_run_ends_(output_run_ends) {
    DCHECK_GT(input_array.length, 0);
  }

  /// \brief Give a pass over the input data and count the number of runs
  ///
  /// \return a tuple with the number of non-null run values, the total number of runs,
  /// and the data buffer size for string and binary types
  ARROW_NOINLINE std::tuple<int64_t, int64_t, int64_t> CountNumberOfRuns() const {
    int64_t read_offset = input_offset_;
    ValueRepr current_run;
    bool current_run_valid = read_write_value_.ReadValue(&current_run, read_offset);
    read_offset += 1;
    int64_t num_valid_runs = current_run_valid ? 1 : 0;
    int64_t num_output_runs = 1;
    int64_t data_buffer_size = 0;
    if constexpr (is_base_binary_like(ValueType::type_id)) {
      data_buffer_size = current_run_valid ? current_run.size() : 0;
    }
    for (; read_offset < input_offset_ + input_length_; read_offset += 1) {
      ValueRepr value;
      const bool valid = read_write_value_.ReadValue(&value, read_offset);

      const bool open_new_run =
          valid != current_run_valid || !read_write_value_.Compare(value, current_run);
      if (open_new_run) {
        // Open the new run
        current_run = value;
        current_run_valid = valid;
        // Count the new run
        num_output_runs += 1;
        num_valid_runs += valid ? 1 : 0;
        if constexpr (is_base_binary_like(ValueType::type_id)) {
          data_buffer_size += valid ? current_run.size() : 0;
        }
      }
    }
    return std::make_tuple(num_valid_runs, num_output_runs, data_buffer_size);
  }

  ARROW_NOINLINE int64_t WriteEncodedRuns() {
    DCHECK(output_run_ends_);
    int64_t read_offset = input_offset_;
    int64_t write_offset = 0;
    ValueRepr current_run;
    bool current_run_valid = read_write_value_.ReadValue(&current_run, read_offset);
    read_offset += 1;
    for (; read_offset < input_offset_ + input_length_; read_offset += 1) {
      ValueRepr value;
      const bool valid = read_write_value_.ReadValue(&value, read_offset);

      const bool open_new_run =
          valid != current_run_valid || !read_write_value_.Compare(value, current_run);
      if (open_new_run) {
        // Close the current run first by writing it out
        read_write_value_.WriteValue(write_offset, current_run_valid, current_run);
        const int64_t run_end = read_offset - input_offset_;
        output_run_ends_[write_offset] = static_cast<RunEndCType>(run_end);
        write_offset += 1;
        // Open the new run
        current_run_valid = valid;
        current_run = value;
      }
    }
    read_write_value_.WriteValue(write_offset, current_run_valid, current_run);
    DCHECK_EQ(input_length_, read_offset - input_offset_);
    output_run_ends_[write_offset] = static_cast<RunEndCType>(input_length_);
    return write_offset + 1;
  }
};

// Allocation and helper functions
ARROW_EXPORT Result<std::shared_ptr<Buffer>> AllocateValuesBuffer(
    int64_t length, const DataType& type, MemoryPool* pool, int64_t data_buffer_size);

ARROW_EXPORT Result<std::shared_ptr<ArrayData>> PreallocateRunEndsArray(
    const std::shared_ptr<DataType>& run_end_type, int64_t physical_length,
    MemoryPool* pool);

ARROW_EXPORT Result<std::shared_ptr<ArrayData>> PreallocateValuesArray(
    const std::shared_ptr<DataType>& value_type, bool has_validity_buffer, int64_t length,
    MemoryPool* pool, int64_t data_buffer_size);

ARROW_EXPORT Result<std::shared_ptr<ArrayData>> PreallocateREEArray(
    std::shared_ptr<RunEndEncodedType> ree_type, bool has_validity_buffer,
    int64_t logical_length, int64_t physical_length, MemoryPool* pool,
    int64_t data_buffer_size);

ARROW_EXPORT void WriteSingleRunEnd(ArrayData* run_ends_data, int64_t run_end);

ARROW_EXPORT Result<std::shared_ptr<ArrayData>> MakeNullREEArray(
    const std::shared_ptr<DataType>& run_end_type, int64_t logical_length,
    MemoryPool* pool);

ARROW_EXPORT Status ValidateRunEndType(const std::shared_ptr<DataType>& run_end_type,
                                       int64_t input_length);

}  // namespace internal
}  // namespace ree_util
}  // namespace arrow
