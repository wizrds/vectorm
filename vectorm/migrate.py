from __future__ import annotations

from collections import defaultdict
from typing import Awaitable, Callable, Sequence

from vectorm.backend import VectorStoreBackend
from vectorm.document import TDocument
from vectorm.filter import FieldExpression
from vectorm.types import FieldIndexParams


class MigrateOp:
    def __init__(self, backend: VectorStoreBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> VectorStoreBackend:
        return self._backend

    async def create_collection(
        self,
        collection_name: str,
        document_type: type[TDocument] | None = None,
        **kwargs,
    ) -> None:
        await self._backend.create_collection(
            collection_name,
            document_type=document_type,
            **kwargs,
        )

    async def delete_collection(
        self,
        collection_name: str,
        **kwargs
    ) -> None:
        await self._backend.delete_collection(
            collection_name=collection_name,
            **kwargs
        )

    async def collection_exists(
        self,
        collection_name: str,
        **kwargs
    ) -> bool:
        return await self._backend.collection_exists(
            collection_name=collection_name,
            **kwargs
        )

    async def create_index(
        self,
        collection_name: str,
        field_name: FieldExpression,
        index_params: FieldIndexParams,
        **kwargs,
    ) -> None:
        await self._backend.create_index(
            collection_name=collection_name,
            field_name=field_name,
            index_params=index_params,
            **kwargs,
        )

    async def delete_index(
        self,
        collection_name: str,
        field_name: FieldExpression,
        **kwargs,
    ) -> None:
        await self._backend.delete_index(
            collection_name=collection_name,
            field_name=field_name,
            **kwargs,
        )

    async def index_exists(
        self,
        collection_name: str,
        field_name: FieldExpression,
        **kwargs,
    ) -> bool:
        return await self._backend.index_exists(
            collection_name=collection_name,
            field_name=field_name,
            **kwargs,
        )


class Revision:
    revision_id: str
    previous_revision_id: str | None = None

    def __repr__(self) -> str:
        return f"<Revision {self.revision_id}>"

    def __str__(self) -> str:
        return self.revision_id

    async def upgrade(self, op: MigrateOp) -> None:
        raise NotImplementedError

    async def downgrade(self, op: MigrateOp) -> None:
        raise NotImplementedError


class RevisionGraph:
    def __init__(self, revisions: Sequence[Revision]) -> None:
        self._children = defaultdict(list)  # revision_id -> list of child revision_ids
        self._parents = {}  # revision_id -> parent revision_id

        for revision in revisions:
            if revision.previous_revision_id:
                self._children[revision.previous_revision_id].append(revision.revision_id)
                self._parents[revision.revision_id] = revision.previous_revision_id

    def find_path(
        self,
        from_revision_id: str,
        to_revision_id: str,
        get_neighbors: Callable[[str], list[str]],
    ) -> list[str] | None:
        if from_revision_id == to_revision_id:
            return [from_revision_id]

        visited = set()
        queue = [(from_revision_id, [from_revision_id])]  # (current_revision_id, current_path)

        while queue:
            current, path = queue.pop(0)

            if current in visited:
                continue

            visited.add(current)

            for neighbor in get_neighbors(current):
                if neighbor == to_revision_id:
                    return path + [neighbor]
                else:
                    queue.append((neighbor, path + [neighbor]))

        return None

    def find_forward_path(
        self,
        from_revision_id: str,
        to_revision_id: str,
    ) -> list[str] | None:
        return self.find_path(
            from_revision_id,
            to_revision_id,
            lambda rev_id: self._children.get(rev_id, []),
        )

    def find_reverse_path(
        self,
        from_revision_id: str,
        to_revision_id: str,
    ) -> list[str] | None:
        return self.find_path(
            from_revision_id,
            to_revision_id,
            lambda rev_id: [self._parents[rev_id]] if rev_id in self._parents else [],
        )


class RevisionChain:
    def __init__(self, revisions: Sequence[Revision]) -> None:
        self._revisions = (
            {revision.revision_id: revision for revision in revisions}
            if revisions
            else {}
        )
        self._graph = RevisionGraph(revisions)
        self._head = self.get_revision_head()
        self._tail = self.get_revision_tail()

    def __repr__(self) -> str:
        return f"<RevisionChain head={self._head} tail={self._tail}>"

    def __str__(self) -> str:
        return f"{self._head} -> ... -> {self._tail}" if self._head and self._tail else "<empty>"

    @property
    def head(self) -> str | None:
        return self._head

    @property
    def tail(self) -> str | None:
        return self._tail

    def get_revision_head(self) -> str | None:
        return next(
            iter(set(self._revisions) - {
                revision.previous_revision_id
                for revision in self._revisions.values()
                if revision.previous_revision_id
            }),
            None
        )

    def get_revision_tail(self) -> str | None:
        return next(
            iter(set(self._revisions) - {
                revision.revision_id
                for revision in self._revisions.values()
                if revision.previous_revision_id
            }),
            None
        )

    def get_revision(self, revision_id: str) -> Revision | None:
        return self._revisions.get(revision_id, None)

    def get_upgrade_path(
        self,
        from_revision_id: str,
        to_revision_id: str,
    ) -> list[Revision] | None:
        path_ids = self._graph.find_forward_path(from_revision_id, to_revision_id)
        if path_ids is None:
            return None

        return [self._revisions[rev_id] for rev_id in path_ids]

    def get_downgrade_path(
        self,
        from_revision_id: str,
        to_revision_id: str,
    ) -> list[Revision] | None:
        path_ids = self._graph.find_reverse_path(from_revision_id, to_revision_id)
        if path_ids is None:
            return None

        return [self._revisions[rev_id] for rev_id in path_ids]


class Migrator:
    def __init__(self, revision_chain: RevisionChain | Sequence[Revision]) -> None:
        self._revision_chain = (
            revision_chain
            if isinstance(revision_chain, RevisionChain)
            else RevisionChain(revision_chain)
        )

    async def apply(
        self,
        backend: VectorStoreBackend,
        target_revision_id: str,
        get_path: Callable[[str, str], list[Revision] | None],
        apply_op: Callable[[MigrateOp, Revision], Awaitable[None]],
        initial_revision_id: str | None = None,
    ) -> None:
        current_revision_id = await backend.current_revision_id()

        if current_revision_id == target_revision_id:
            return

        from_revision_id = current_revision_id or initial_revision_id
        if from_revision_id is None:
            raise ValueError("Cannot apply migration without a starting revision.")

        path = get_path(from_revision_id, target_revision_id)
        if path is None:
            raise ValueError(
                "No path found from revision "
                f"{from_revision_id} to {target_revision_id}."
            )

        op = MigrateOp(backend)
        for revision in (path[1:] if current_revision_id is not None else path):
            await apply_op(op, revision)
            await backend.set_revision_id(revision.revision_id)

    async def upgrade(
        self,
        backend: VectorStoreBackend,
        revision_id: str = "head",
    ) -> None:
        await self.apply(
            backend=backend,
            target_revision_id=(
                self._revision_chain.head
                if revision_id == "head" and self._revision_chain.head is not None
                else revision_id
            ),
            get_path=self._revision_chain.get_upgrade_path,
            apply_op=lambda op, rev: rev.upgrade(op),
            initial_revision_id=self._revision_chain.tail,
        )

    async def downgrade(
        self,
        backend: VectorStoreBackend,
        revision_id: str = "tail",
    ) -> None:
        await self.apply(
            backend=backend,
            target_revision_id=(
                self._revision_chain.tail
                if revision_id == "tail" and self._revision_chain.tail is not None
                else revision_id
            ),
            get_path=self._revision_chain.get_downgrade_path,
            apply_op=lambda op, rev: rev.downgrade(op),
            initial_revision_id=self._revision_chain.head,
        )
