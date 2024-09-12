from typing import Iterable, List, NamedTuple, Optional, Tuple, overload

from vllm.sequence import Sequence


class TokenRangeAnnotation(NamedTuple):
    """
    Annotates a range of placeholder tokens to capture content that will
    replace them.
    """

    content_hash: int
    content_offset: int
    token_index: int
    token_count: int

    @property
    def end_index(self) -> int:
        return self.token_index + self.token_count

    @staticmethod
    def are_adjacent(left: "TokenRangeAnnotation",
                     right: "TokenRangeAnnotation") -> bool:
        """
        Indicates whether two annotations represent adjacent ranges in the
        hashed content.
        """

        return (left.content_hash == right.content_hash and
                left.content_offset + left.token_count == right.content_offset)

    def adjusted(self, tokens_start: int,
                 tokens_end: int) -> Optional["TokenRangeAnnotation"]:
        """
        Computes a new TokenRangeAnnotation that corresponds to the same
        content in a slice of the original token IDs.
        """

        unclamped_annotation_start = self.token_index - tokens_start
        unclamped_annotation_end = unclamped_annotation_start + self.token_count

        annotation_start = max(0, unclamped_annotation_start)
        annotation_end = min(tokens_end - tokens_start,
                             unclamped_annotation_end)

        if annotation_start >= annotation_end:
            # There is no overlap.
            return None

        return TokenRangeAnnotation(
            content_hash=self.content_hash,
            content_offset=self.content_offset + annotation_start -
            unclamped_annotation_start,
            token_index=annotation_start,
            token_count=annotation_end - annotation_start)


class TokenIds:
    token_ids: Tuple[int, ...]
    annotations: Tuple[TokenRangeAnnotation, ...]

    def __init__(self,
                 token_ids: Iterable[int] = (),
                 annotations: Iterable[TokenRangeAnnotation] = ()):
        self.token_ids = tuple(token_ids)
        self.annotations = tuple(annotations)

        # Ensure that the token annotations are monotonic.
        current_token_index = 0
        for annotation in self.annotations:
            if (annotation.token_index < current_token_index
                    or annotation.token_count < 0):
                raise ValueError("TokenRangeAnnotations must be sorted and "
                                 "non-overlapping.")

            current_token_index = annotation.end_index

        if current_token_index > len(self.token_ids):
            raise ValueError("TokenRangeAnnotations must be entirely "
                             "contained within the token IDs.")

    @staticmethod
    def from_sequence(sequence: Sequence, offset: int) -> "TokenIds":
        # Since the block table is append-only, the unseen token ids are the
        # ones after the appended ones.
        token_ids = sequence.get_token_ids()[offset:]

        # Adjust any annotations for the new token ids list.
        adjusted_annotations = (a.adjusted(offset, offset + len(token_ids))
                                for a in sequence.token_annotations)
        filtered_annotations = (a for a in adjusted_annotations
                                if a is not None)
        sorted_annotations = sorted(filtered_annotations,
                                    key=lambda a: a.token_index)
        return TokenIds(token_ids, sorted_annotations)

    def chunks(self,
               chunk_size: int,
               *,
               first_chunk_size: Optional[int] = None):
        """
        Yields successive chunks over the TokenIds, taking care to filter
        or split TokenRangeAnnotations accordingly.
        """

        current_annotation_index = 0
        i = 0
        current_chunk_size = (chunk_size if first_chunk_size is None else
                              first_chunk_size)

        while i < len(self.token_ids):
            current_chunk_annotations: List[TokenRangeAnnotation] = []
            while current_annotation_index < len(self.annotations):
                existing_annotation = self.annotations[
                    current_annotation_index]
                if existing_annotation.token_index >= i + current_chunk_size:
                    # This annotation starts after the current chunk.
                    break

                # Create a new annotation.
                new_annotation = existing_annotation.adjusted(
                    tokens_start=i, tokens_end=i + current_chunk_size)
                assert new_annotation is not None, (
                    "The existing annotation should overlap with the new one.")
                current_chunk_annotations.append(new_annotation)
                if (i + new_annotation.end_index ==
                        existing_annotation.end_index):
                    # We've used up this annotation.
                    current_annotation_index += 1
                else:
                    break

            yield TokenIds(self.token_ids[i:i + current_chunk_size],
                           current_chunk_annotations)

            i += current_chunk_size
            current_chunk_size = chunk_size

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TokenIds):
            return (self.token_ids == other.token_ids
                    and self.annotations == other.annotations)

        return NotImplemented

    def __add__(self, other: "TokenIds") -> "TokenIds":
        """
        Combines two ``TokenIds``, possibly merging ``TokenRangeAnnotion``s.

        ``TokenRangeAnnotation``s at the boundary will be coalesced into a
        single annotation if they have the same content hash and they cover
        adjacent portions of the hashed content.
        """

        if not self.token_ids:
            return other
        elif not other.token_ids:
            return self

        # Merge the token annotations if necessary
        if not other.annotations:
            combined_annotations: Iterable[
                TokenRangeAnnotation] = self.annotations
        else:
            combined_annotations = list(self.annotations)
            for annotation in other.annotations:
                if combined_annotations:
                    # Check if we can coalesce this annotation with the last.
                    last_annotation = combined_annotations[-1]
                    if (TokenRangeAnnotation.are_adjacent(
                            last_annotation, annotation) and
                            last_annotation.end_index == len(self.token_ids)
                            and annotation.token_index == 0):
                        combined_annotations[-1] = TokenRangeAnnotation(
                            content_hash=last_annotation.content_hash,
                            content_offset=last_annotation.content_offset,
                            token_index=last_annotation.token_index,
                            token_count=last_annotation.token_count +
                            annotation.token_count)
                        continue

                combined_annotations.append(
                    TokenRangeAnnotation(
                        content_hash=annotation.content_hash,
                        content_offset=annotation.content_offset,
                        token_index=len(self.token_ids) +
                        annotation.token_index,
                        token_count=annotation.token_count))

        return TokenIds(token_ids=self.token_ids + other.token_ids,
                        annotations=combined_annotations)

    def __len__(self) -> int:
        return len(self.token_ids)

    @overload
    def __getitem__(self, key: int) -> int:
        ...

    @overload
    def __getitem__(self, key: slice) -> "TokenIds":
        ...

    def __getitem__(self, key):
        """
        Gets a single token at an index or a slice of ``TokenIds``.
        """
        if isinstance(key, int):
            return self.token_ids[key]

        if isinstance(key, slice):
            if key.step:
                raise IndexError("Step is not supported.")

            # Resolve negative indices.
            if key.start is None:
                start = 0
            elif key.start < 0:
                start = len(self) + key.start
            else:
                start = key.start

            if key.stop is None:
                stop = len(self)
            elif key.stop < 0:
                stop = len(self) + key.stop
            else:
                stop = key.stop

            # Clamp the indices.
            start = max(0, min(len(self), start))
            stop = max(start, min(len(self), stop))

            chunks_iter = iter(
                self.chunks(chunk_size=stop - start, first_chunk_size=start))

            # Drop the first chunk and return the second chunk.
            try:
                next(chunks_iter)
                return next(chunks_iter)
            except StopIteration:
                return TokenIds()

        raise TypeError(f"Unsupported key type: {type(key)}")
