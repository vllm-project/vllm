from typing import Iterable, List, NamedTuple, Optional, Tuple, overload


class TokenRangeAnnotation(NamedTuple):
    """
    Annotates a range of placeholder tokens to capture content that will
    replace them.
    """

    content_hash: int
    content_offset: int
    token_start_index: int
    token_end_index: int

    @property
    def token_count(self) -> int:
        return self.token_end_index - self.token_start_index

    @staticmethod
    def are_adjacent(left: "TokenRangeAnnotation",
                     right: "TokenRangeAnnotation") -> bool:
        """
        Indicates whether two annotations represent adjacent ranges in the
        hashed content.
        """

        return (left.content_hash == right.content_hash and
                left.content_offset + left.token_count == right.content_offset)

    def clipped_to_slice(self, tokens_start: int,
                         tokens_end: int) -> Optional["TokenRangeAnnotation"]:
        """
        Computes a new TokenRangeAnnotation that corresponds to the same
        content in a slice of the original token IDs.

        For example, consider the following token IDs/annotations:

        AAAA BBBB What do these images have in common?
        
        A = TokenRangeAnnotation(0xA, 0, 0, 4)
        B = TokenRangeAnnotation(0xB, 0, 5, 9)

        tokens = AAAA BBBB What do these images have in common?
                [AAAA]

        A.clipped_to_slice(0, 4) = TokenRangeAnnotation(0xA, 0, 0, 4)
        B.clipped_to_slice(0, 4) = None

        tokens = AAAA BBBB What do these images have in common?
                  [AA BB]

        A.clipped_to_slice(2, 7) = TokenRangeAnnotation(0xA, 2, 0, 2)
        B.clipped_to_slice(2, 7) = TokenRangeAnnotation(0xB, 0, 3, 5)

        tokens = AAAA BBBB What do these images have in common?
                     [BBBB What]

        A.clipped_to_slice(5, 14) = None
        B.clipped_to_slice(5, 14) = TokenRangeAnnotation(0xB, 0, 0, 4)
        """

        unclamped_annotation_start = self.token_start_index - tokens_start
        unclamped_annotation_end = self.token_end_index - tokens_start

        annotation_start = max(0, unclamped_annotation_start)
        annotation_end = min(tokens_end - tokens_start,
                             unclamped_annotation_end)

        if annotation_start >= annotation_end:
            # There is no overlap.
            return None

        return TokenRangeAnnotation(content_hash=self.content_hash,
                                    content_offset=self.content_offset +
                                    annotation_start -
                                    unclamped_annotation_start,
                                    token_start_index=annotation_start,
                                    token_end_index=annotation_end)


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
            if (annotation.token_start_index < current_token_index or
                    annotation.token_end_index < annotation.token_start_index):
                raise ValueError("TokenRangeAnnotations must be sorted and "
                                 "non-overlapping.")

            current_token_index = annotation.token_end_index

        if current_token_index > len(self.token_ids):
            raise ValueError("TokenRangeAnnotations must be entirely "
                             "contained within the token IDs.")

    def to_chunks(self,
                  chunk_size: int,
                  *,
                  first_chunk_size: Optional[int] = None):
        """
        Yields successive chunks over the TokenIds, taking care to filter
        or split TokenRangeAnnotations accordingly.
        """

        current_annotation_index = 0
        current_chunk_start = 0
        current_chunk_end = (chunk_size
                             if first_chunk_size is None else first_chunk_size)

        while current_chunk_start < len(self.token_ids):
            current_chunk_annotations: List[TokenRangeAnnotation] = []
            while current_annotation_index < len(self.annotations):
                existing_annotation = self.annotations[
                    current_annotation_index]
                if existing_annotation.token_start_index >= current_chunk_end:
                    # This annotation starts after the current chunk.
                    break

                # Create a new annotation.
                new_annotation = existing_annotation.clipped_to_slice(
                    tokens_start=current_chunk_start,
                    tokens_end=current_chunk_end)
                assert new_annotation is not None, (
                    "The existing annotation should overlap with the new one.")
                current_chunk_annotations.append(new_annotation)
                if (current_chunk_start + new_annotation.token_end_index ==
                        existing_annotation.token_end_index):
                    # We've used up this annotation.
                    current_annotation_index += 1
                else:
                    break

            yield TokenIds(
                self.token_ids[current_chunk_start:current_chunk_end],
                current_chunk_annotations)

            current_chunk_start = current_chunk_end
            current_chunk_end = current_chunk_start + chunk_size

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
                            last_annotation, annotation)
                            and last_annotation.token_end_index == len(
                                self.token_ids)
                            and annotation.token_start_index == 0):
                        combined_annotations[-1] = TokenRangeAnnotation(
                            content_hash=last_annotation.content_hash,
                            content_offset=last_annotation.content_offset,
                            token_start_index=last_annotation.
                            token_start_index,
                            token_end_index=last_annotation.token_end_index +
                            annotation.token_count)
                        continue

                combined_annotations.append(
                    TokenRangeAnnotation(
                        content_hash=annotation.content_hash,
                        content_offset=annotation.content_offset,
                        token_start_index=len(self.token_ids) +
                        annotation.token_start_index,
                        token_end_index=len(self.token_ids) +
                        annotation.token_end_index))

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
            start = key.start or 0
            start += len(self) if start < 0 else 0

            stop = key.stop if key.stop is not None else len(self)
            stop += len(self) if stop < 0 else 0

            # Fast path for the common case where the new slice doesn't
            # include any annotations (e.g. slicing a decoded token).
            if (not self.annotations
                    or start >= self.annotations[-1].token_end_index
                    or stop <= self.annotations[0].token_start_index):
                return TokenIds(self.token_ids[start:stop])

            # Clamp the indices.
            start = max(0, min(len(self), start))
            stop = max(start, min(len(self), stop))

            chunks_iter = iter(
                self.to_chunks(chunk_size=stop - start,
                               first_chunk_size=start))

            # Drop the first chunk and return the second chunk.
            try:
                next(chunks_iter)
                return next(chunks_iter)
            except StopIteration:
                return TokenIds()

        raise TypeError(f"Unsupported key type: {type(key)}")
