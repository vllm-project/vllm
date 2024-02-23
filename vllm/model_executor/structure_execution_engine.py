# structure_execution_engine.py
# Copyright 2024 DeepInfra, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations
from typing import Any, Optional, Union, Tuple, Dict, cast, Iterable, overload
from typing_extensions import TypeAlias
import sys

VERBOSE: int = 0  # 2 or higher: Print information while constructing the graph

LinkedChars: TypeAlias = Union[Tuple[str, 'LinkedChars'], Tuple[()]]
StructureNodeKey: TypeAlias = Tuple[str, int, str]
SingleNodeKey: TypeAlias = Union[Tuple[str, int, str], Tuple[str, int]]
NodeKeyTuple: TypeAlias = Tuple[SingleNodeKey, ...]
OpTuple: TypeAlias = Tuple[Union[str, StructureNodeKey, None], ...]
PrecalculatedRawGraph: TypeAlias = Dict[NodeKeyTuple, Dict[str, Tuple[OpTuple, SingleNodeKey]]]


def trange(a: str | int | bytes, b_incl: str | int | bytes) -> range:
    return range(
        ord(a) if isinstance(a, (str, bytes)) else a,
        1 + ord(b_incl) if isinstance(b_incl, (str, bytes)) else 1 + b_incl)


class ParserStructureStack:
    """
    Internal stack holding references to StructureNode. This is only used during the initial generation of the raw graph.

    Builds a list of "operations" (push, pop and append_char) into `ops`
    """

    def __init__(self, root_structure: 'Structure') -> None:
        self.ops: list[Optional['StructureNode' | str]] = [root_structure]
        self.stack: list[Any] = [root_structure]

    def __repr__(self) -> str:
        return "<ParserStructureStack " + repr(self.stack) + " result:" + repr(self.ops) + ">"

    def append_char(self, char: str) -> None:
        """
        Output characters are represented as a string in the ops list.
        """
        if len(self.stack) and self.stack[-1].expr.is_string:
            # First case is an optimization, but makes some of the graph search stuff later on more difficult.
            if False and self.ops and type(self.ops[-1]) is str:
                self.ops[-1] += char
            else:
                self.ops.append(char)

    def success(self) -> None:
        self.ops.append('')

    def push(self, structure_node: 'StructureNode') -> None:
        """
        Pushing a structure node is done by appending that structure node to ops.
        """
        assert (not self.stack or not self.stack[-1].expr.is_string)
        self.ops.append(structure_node)
        self.stack.append(structure_node)

    def pop(self, exception_data: Any = None) -> StructureNode:
        """
        Pop is represented as a None in the list of ops.
        """
        self.ops.append(None)
        if len(self.stack) == 0:
            raise Exception(exception_data)
        return self.stack.pop()


class ParserNode:
    """
    Internal base class for a parser node within a structure. ParserNodes are only used when building the precalculated raw graph.
    """

    def __repr__(self) -> str:
        return ('<%s(%s)%r>' % (self.__class__.__name__, repr(self.expr), self.path) +
                "@" + self.structure.name + "/" + str(self.structure.node_to_index.get(self, None)))

    def __init__(self, expr: Any) -> None:
        self.expr = expr
        self.structure: Structure
        self.node_key: SingleNodeKey
        self.index = -1
        self.reachable: bool = False

    def copy(self) -> ParserNode:
        """
        Clones this node and returns the duplcate.
        """
        raise NotImplementedError("Subclass missing copy")

    def generate_node_key(self) -> SingleNodeKey:
        """
        Returns a key representing this node, which will be used to represent this node in the generated raw graph.
        """
        return self.structure.name, self.index

    @property
    def path(self) -> tuple:
        """
        Returns the path to this parser node within the containing Structure
        """
        return self.structure.get_path(self)

    def parent(self) -> ParserNode:
        """
        Returns the parent parser node, or null if this is the root Structure
        """
        path = self.path
        assert path
        # if not path:
        #     return self
        if len(path) == 1:
            return self.structure
        return self.structure.path_to_node(path[:-1])

    def which_match(self, char: str) -> int:
        """
        Returns a node-specific match index for this character, or 0 if no match was found.
        """
        raise NotImplementedError()

    def next(self, char: Optional[str], stack: Optional[ParserStructureStack], which_match: int, success: bool = False, from_child: Any = -1) -> Optional[ParserNode]:
        """
        Returns the next node in the parse, given the output of which_match. When which_match is 0, proceeds to the parent node, passing success and our path as from_child.
        """
        if stack and which_match != 0:
            assert from_child == -1 and char is not None
            stack.append_char(char)
        par = self.parent()
        if par:
            return par.next(None, stack, 0, success or which_match != 0, self.path[-1])
        return None

    def n(self, char: str, stack: Optional[ParserStructureStack]) -> Optional[ParserNode]:
        """
        Helper method to invoke which_match followed by next.
        """
        return self.next(char, stack, self.which_match(char))


class ConstantNode(ParserNode):
    """
    Parser node representing a single character, either an exact match, a range, or a tuple of possible choices.
    """

    def __init__(self, expr: Any) -> None:
        super().__init__(expr)

    def copy(self) -> ParserNode:
        return ConstantNode(self.expr)

    def __repr__(self) -> str:
        return repr(self.expr) + "@" + self.structure.name + "/" + str(self.structure.node_to_index.get(self, None))

    def next(self, char: Optional[str], stack: Optional[ParserStructureStack], which_match: int, success: bool = False, from_child: Any = -1) -> Optional[ParserNode]:
        if stack and which_match != 0:
            assert from_child == -1 and char is not None
            stack.append_char(char)
            par = self.parent()
            if par:
                return par.next(None, stack, 0, success or which_match != 0, self.path[-1])
        return None

    def generate_node_key(self) -> SingleNodeKey:
        if type(self.expr) is str:
            return self.structure.name, self.index, self.expr
        else:
            return self.structure.name, self.index

    def which_match(self, char: str) -> int:
        if type(self.expr) is str:
            return 1 if char == self.expr else 0
        if type(self.expr) is range:
            return 1 if (0 if char == '' else ord(char)) in self.expr else 0
        if type(self.expr) is tuple:
            return 1 if char in self.expr else 0
        return 0


class RegexRangeNode(ConstantNode):
    """
    Parser node representing a single character in the form of a regex [a-z] style range of choices.
    """

    def __init__(self, expr: Any) -> None:
        assert (type(expr) is str and expr[0] == '[' and expr[-1] == ']' and len(expr) > 2)
        self.invert = expr[1] == '^'
        i = self.invert + 1
        self.ranges = []
        self.chars = []
        while i + 1 < len(expr):
            if i + 3 < len(expr) and expr[i + 1] == '-':
                self.ranges.append(trange(ord(expr[i]), ord(expr[i + 2])))
                i += 3
            else:
                self.chars.append(expr[i])
                i += 1
        super().__init__(expr)

    def __repr__(self) -> str:
        return repr(self.expr) + "@" + self.structure.name + "/" + str(self.structure.node_to_index.get(self, None))

    def copy(self) -> ParserNode:
        return RegexRangeNode(self.expr)

    def generate_node_key(self) -> SingleNodeKey:
        return self.structure.name, self.index

    def which_match(self, char: str) -> int:
        if not char:
            return 0
        if char in self.chars or any(ord(char) in r for r in self.ranges):
            return 1 - self.invert
        else:
            return 0 + self.invert


class StructureNode(ConstantNode):
    """
    Parser node which contains a reference to another Structure: This node will push itself onto the structure stack.
    """

    def __init__(self, expr: "Structure") -> None:
        super().__init__(expr)

    def copy(self) -> ParserNode:
        return StructureNode(self.expr)

    def generate_node_key(self) -> SingleNodeKey:
        return self.structure.name, self.index, '' + self.expr.name

    def __repr__(self) -> str:
        return self.expr.name + "@" + self.structure.name + "/" + str(self.structure.node_to_index.get(self, None))

    def which_match(self, char: str) -> int:
        return self.expr.root_node.which_match(char)

    def next(self, char: Optional[str], stack: Optional[ParserStructureStack], which_match: int, success: bool = False, from_child: Any = -1) -> Optional[ParserNode]:
        if which_match != 0:
            if stack is None:
                return self
            stack.push(self)
            return self.expr.root_node.next(char, stack, which_match)
        return super().next(char, stack, which_match, success, from_child)


class Sequence(ParserNode):
    """
    Parser node which contains a list of children. For example a word will be represented as a sequence of ConstantNode.
    """

    def __init__(self, expr: Any) -> None:
        super().__init__(expr)

    def copy(self) -> ParserNode:
        return Sequence(self.expr)

    def which_match(self, char: str) -> int:
        return self.expr[0].which_match(char)

    def next(self, char: Optional[str], stack: Optional[ParserStructureStack], which_match: int, success: bool = False, from_child: int = -1) -> Optional[ParserNode]:
        if which_match != 0:
            return self.expr[0].next(char, stack, which_match)
        assert (which_match == 0)
        if not success:
            return None
        if from_child != -1:
            last_finished_idx = from_child
        else:
            last_finished_idx = -1
        new_idx = last_finished_idx + 1
        if new_idx >= len(self.expr) or not success:
            return super().next(None, stack, 0, success, self.path[-1])
        return self.expr[new_idx]


class AnyOf(ParserNode):
    """
    Parser node which will proceed to one of its children, depending on the first child to match.
    """

    def __init__(self, expr: Any) -> None:
        super().__init__(expr)

    def copy(self) -> ParserNode:
        return AnyOf(self.expr)

    def which_match(self, char: str) -> int:
        for i, sub_node in enumerate(self.expr):
            if sub_node.which_match(char) > 0:
                return i + 1
        return 0

    def next(self, char: Optional[str], stack: Optional[ParserStructureStack], which_match: int, success: bool = False, from_child: int = -1) -> Optional[ParserNode]:
        if which_match == 0:
            if not success:
                return None
            return super().next(None, stack, 0, success, self.path[-1])
        else:
            assert (char is not None)
            next_node: ParserNode = self.expr[which_match - 1]
            which_match = next_node.which_match(char)
            return next_node.next(char, stack, which_match)


class ZeroOrMore(ParserNode):
    """
    Parser node which optionally continues to its child if it matches, otherwise proceeds to its parent without failing.
    """

    def __init__(self, expr: Any, only_once: bool = False) -> None:
        super().__init__(expr)
        self.only_once = only_once

    def copy(self) -> ParserNode:
        return ZeroOrMore(self.expr)

    def which_match(self, char: str) -> int:
        if char == '':
            return 2
        if self.expr.which_match(char) != 0:
            return 1
        nxt = super().next(None, None, 0, True, self.path[-1])
        if nxt is not None and nxt.which_match(char) != 0:
            return 2
        return -1

    def next(self, char: Optional[str], stack: Optional[ParserStructureStack], which_match: int, success: bool = False, from_child: str = '') -> Optional[ParserNode]:
        if which_match == 1:
            return self.expr.next(char, stack, self.expr.which_match(char))
        elif which_match == -1 or which_match == 2:
            assert from_child == '' and char is not None
            next_node = super().next(None, stack, 0, True, from_child=self.path[-1])
            if next_node is None:
                return super().next(None, stack, 0, False, from_child=self.path[-1])
            if from_child != '':
                return next_node
            if isinstance(next_node, StructureNode):
                return next_node
            which_match = next_node.which_match(char)
            return next_node.next(char, stack, which_match)
        elif success and not self.only_once:
            assert (from_child == self.__class__.__name__)
            return self
        else:
            assert (from_child == self.__class__.__name__)
            return super().next(None, stack, 0, success, from_child=self.path[-1])


# Ensuring the correct amount of whitespace must be handled by the implementation.
class AutoIndent(ZeroOrMore):
    """
    ZeroOrMore space node, with a hint in its node key to force the correct amount of indentation.
    """

    def copy(self) -> ParserNode:
        return AutoIndent(self.expr)

    def generate_node_key(self) -> SingleNodeKey:
        return self.structure.name, self.index, 'indent'


# Ensuring the correct amount of whitespace must be handled by the implementation.
class AutoIndentEnd(AutoIndent):
    def copy(self) -> ParserNode:
        return AutoIndentEnd(self.expr)

    def generate_node_key(self) -> SingleNodeKey:
        return self.structure.name, self.index, 'indent_end'


class EndNode(ParserNode):
    """
    Represents the end state of a Structure. The parser will rest here after parsing a Structure's expression.
    The parser is greedy and cannot proceed to the parent structure unless it knows which character to consume next.
    """

    def __init__(self) -> None:
        super().__init__(None)

    def __repr__(self) -> str:
        return 'EndNode(%r)' % (self.structure.name,)

    def copy(self) -> ParserNode:
        return EndNode()

    def generate_node_key(self) -> SingleNodeKey:
        return self.structure.name, self.index, 'end'

    def which_match(self, char: str) -> int:
        if char == '':
            return 2
        return -1

    def next(self, char: Optional[str], stack: Optional[ParserStructureStack], which_match: int, success: bool = False, from_child: str = '') -> Optional[ParserNode]:
        return self.parent()
            

class OptionalNode(ZeroOrMore):
    """
    Optional parser node similar to ZeroOrMore, but does not loop back to itself.
    """

    def __init__(self, expr: Any) -> None:
        super().__init__(expr, only_once=True)

    def copy(self) -> ParserNode:
        return OptionalNode(self.expr)


PathType = Union[Tuple[()], Tuple[Union[int, str], ...]]


class Structure(StructureNode):
    """
    A Structure, containing an expression, the list of recursive node paths by index, and either producing a string or containing children.
    """

    def __init__(self, name: str) -> None:
        super().__init__(self)
        self.structure = self
        self.name: str = name
        self.end_structure: EndNode = EndNode()
        self.end_structure.structure = self
        self.end_structure.reachable = True
        self.end_structure.index = 0
        self.is_string: bool = True
        self.indices: list[tuple[ParserNode, PathType]] = []
        self.path_to_index: dict[PathType, int] = {}
        self.node_to_index: dict[ParserNode, int] = {}
        # Undefined:
        self.root_node: ParserNode
        self.graph: NodeStructureGraph

    def generate_node_key(self) -> SingleNodeKey:
        return self.structure.name, self.index, 'root'

    def init(self, main_graph: NodeStructureGraph) -> 'Structure':
        self.graph = main_graph
        main_graph.add_structure(self)
        return self

    def walk_and_replace(self, arg: Any, graph_structures) -> ParserNode:
        ret: ParserNode
        if isinstance(arg, type(self)):
            ret = StructureNode(graph_structures[arg.name])
        elif isinstance(arg, ConstantNode):
            arg.structure = self
            ret = arg.copy()
        elif isinstance(arg, ParserNode):
            ret = arg.copy()
            ret.expr = self.walk_and_replace(ret.expr, graph_structures)
            if isinstance(ret.expr, StructureNode):
                self.is_string = False
        elif isinstance(arg, list) or (isinstance(arg, str) and len(arg) > 1):
            if isinstance(arg, str):
                self.graph.special_words.add(arg)
            ret = Sequence([self.walk_and_replace(child, graph_structures) for child in arg])
        elif isinstance(arg, tuple) and all(
                (isinstance(x, str) and len(x) <= 1) or
                (isinstance(x, ConstantNode) and isinstance(x.expr, str) and len(x.expr) <= 1) for x in arg):
            ret = ConstantNode(arg)
        elif isinstance(arg, tuple) or (isinstance(arg, str) and len(arg) > 1):
            ret = AnyOf(tuple([self.walk_and_replace(child, graph_structures) for child in arg]))
        elif isinstance(arg, Structure):
            ret = StructureNode(arg)
        else:
            ret = ConstantNode(arg)
        ret.structure = self
        if isinstance(ret, StructureNode):
            self.is_string = False
        return ret

    def construct_indices(self) -> None:
        self.indices.append((self.end_structure, ('End',)))
        self.root_node.reachable = True
        self._construct_indices_recursive(self.root_node, ('Root',))
        self.indices.append((self, ()))
        self.index = len(self.indices) - 1

    def _construct_indices_recursive(self, node: Any, path: PathType, optional_root: Optional[ParserNode] = None) -> None:
        idx = len(self.indices)
        if isinstance(node, Structure):
            pass
        elif isinstance(node, ParserNode):
            if optional_root and not isinstance(node, StructureNode):
                self.indices.append((node, path))
                node.index = idx
                node.reachable = False
            else:
                self.indices.append((node, path))
                node.index = idx
                node.reachable = True

            if isinstance(node, (AnyOf, OptionalNode)):
                optional_root = node
            elif isinstance(node, ZeroOrMore):
                optional_root = None

            if node.expr and isinstance(node.expr, (tuple, list)):
                for i, child in enumerate(node.expr):
                    if i != 0 and isinstance(node, Sequence):
                        optional_root = None
                    self._construct_indices_recursive(child, path + (i,), optional_root=optional_root)
            elif node.expr:
                self._construct_indices_recursive(node.expr, path + (type(node).__name__,), optional_root=optional_root)

    def get_path(self, node: ParserNode) -> tuple:
        return self.indices[self.node_to_index[node]][1]

    def path_to_node(self, path: PathType) -> ParserNode:
        return self.indices[self.path_to_index[path]][0]

    def index_to_node(self, idx: int) -> ParserNode:
        return self.indices[idx][0]

    def index_to_path(self, idx: int) -> PathType:
        return self.indices[idx][1]

    def which_match(self, c: str) -> int:
        if c == '':
            return 1
        return 0

    def next(self, char: Optional[str], stack: Optional[ParserStructureStack], which_match: int, success: bool = False, from_child: str = '') -> Optional[ParserNode]:
        if which_match == 1:
            return None
        if success and from_child == 'Root':
            return self.end_structure
        if stack is None:
            return None
        last_exp = stack.pop(self)
        if type(last_exp) is Structure:
            assert last_exp
            return last_exp
        assert (type(last_exp) is StructureNode)
        assert (last_exp.expr == self)
        exp = last_exp.parent()
        if exp is not None:
            return exp.next(None, stack, 0, True, last_exp.path[-1])
        return None

    def tick(self, structure_stack: ParserStructureStack, exp_in: ParserNode, c: str) -> Optional[ParserNode]:
        if exp_in == self:
            return None
        exp: Optional[ParserNode] = exp_in.n(c, structure_stack)
        if exp is None:
            if c == '':
                structure_stack.success()
            return None
        else:
            # A grammar is not supposed to have a specific character that forces the parser into a StructureNode.
            # Here are some reasons for this design limitation:
            # 1. There would be no way to distinguish entering into a StructureNode and exiting from one.
            # 2. We never traverse on '': next(c='', success=true)
            # 3. It is undesirable to stop before we have pushed the StructureNode onto the structure_stack ops
            # (since we rely on ops in StructureOpGraph to select the correct token greedily)
            if c == '' and exp != self:
                print("c: " + str(c) + " exp: " + str(exp) + " self: " + str(self))
            assert (c != '' or exp == self)
            return exp

    # Not used in production, just for testing
    def match(self, inp: str) -> Any:  # tuple[Any, Optional[ParserNode]]:
        structure_stack: ParserStructureStack = ParserStructureStack(self)
        exp: Optional[ParserNode] = self.root_node
        for char in inp:
            if exp is None:
                return None
            exp = self.tick(structure_stack, exp, char)
        if exp is None:
            return None
        self.tick(structure_stack, exp, '')
        return structure_stack.ops

    def __repr__(self) -> str:
        return 'Structure(%r)' % (self.name,)

    def __str__(self) -> str:
        return 'Structure(%r, %r)' % (self.name, self.root_node)


class NodeStructureGraph:
    """
    A graph representing a grammar containing a series of parser nodes, which can reference each other.
    """
    def __init__(self) -> None:
        self.structures: dict[str, Structure] = {}
        self.special_words: set[str] = set()
        self.root: Structure

    def add_structure(self, structure: Structure) -> None:
        structure.graph = self
        self.structures[structure.name] = structure

    def init_structure(self, structure: Structure, root_node: Any):
        structure.root_node = structure.walk_and_replace(root_node, self.structures)

    def init_graph(self) -> None:
        for structure in self.structures.values():
            structure.construct_indices()
            idx: int
            node: ParserNode
            path: PathType
            for idx, (node, path) in enumerate(structure.indices):
                structure.path_to_index[path] = idx
                structure.node_to_index[node] = idx
                node.node_key = node.generate_node_key()

    def get_all_basic_entry_points(self) -> list[ParserNode]:
        """
        Returns a list of all possible entry points across all structures.
        Not all nodes are reachable, for example a ConstantNode inside an OptionalNode since it will rest at the OptionalNode.
        """
        entry_points: list[ParserNode] = []
        structure: Structure
        idx: int
        node: ParserNode
        path: PathType
        for structure in self.structures.values():
            for idx, (node, path) in enumerate(structure.indices):
                if node.reachable:
                    entry_points.append(node)
                else:
                    # Skips about 40% of entry points
                    if VERBOSE >= 2:
                        print(structure.name, "Skipping", idx, node, path)
                    pass
        if VERBOSE >= 1:
            print("Found", len(entry_points), "entry points")
        return entry_points

    def calculate_structure_graph(self, token_strings: list[str]) -> PrecalculatedRawGraph:
        """
        Performs the calculation of what is called a PrecalculatedRawGraph.
        Keys in the raw graph are state key tuples, such as (current_state_key,) or (parent_structure_node_key, current_state_key,) or so on.
        Values in the raw graph are a dictionary from char to a tuple of operation list and the next state key.

        This function is quite slow (often 5-10 seconds for JSON), and perhaps should be written in a more optimized language
        """

        def str_to_ll(s: str, i: int = 0) -> LinkedChars:
            if i >= len(s):
                return ()
            return s[i], str_to_ll(s, i + 1)

        def ll_to_str(ll: LinkedChars) -> str:
            ret = ''
            while ll:
                ret += ll[0]
                ll = ll[1]
            return ret

        if VERBOSE >= 1:
            print("Calculating structure graph")
        token_char_lists: list[LinkedChars] = [str_to_ll(s) for s in token_strings]
        tokens_to_process: list[tuple[int, str]] = []

        for token_id, token_string in enumerate(token_strings):
            tokens_to_process.append((token_id, token_string.strip()))

        basic_entry_points: list[ParserNode] = self.get_all_basic_entry_points()

        entry_point_parents: dict[StructureNode, list[tuple[ParserNode, StructureNode]]] = {}
        for outer_structure_node in basic_entry_points:
            if isinstance(outer_structure_node, StructureNode) and not isinstance(outer_structure_node, Structure):
                inner_structure: Structure = outer_structure_node.expr
                if inner_structure not in entry_point_parents:
                    entry_point_parents[inner_structure] = []
                par: ParserNode = outer_structure_node.parent()
                if par is Structure:
                    print("parent failed: " + str(outer_structure_node))
                    continue
                structure_stack: ParserStructureStack = ParserStructureStack(outer_structure_node.structure)
                structure_stack.stack[:] = [outer_structure_node.structure]
                structure_stack.ops.clear()
                result_exp: Optional[ParserNode] = par.next(None, structure_stack, 0, True, outer_structure_node.path[-1])
                assert (result_exp is not None)
                entry_point_parents[inner_structure].append((result_exp, outer_structure_node))

        # Key
        precalculated_raw_graph: PrecalculatedRawGraph = {}
        tokens_calculated: dict[tuple[SingleNodeKey, str], set[tuple[OpTuple, StructureNode, tuple[ParserNode, LinkedChars]]]] = {}
        token_queue: list[tuple[ParserNode, LinkedChars]] = []
        token_char_list: LinkedChars
        for node in basic_entry_points:
            if VERBOSE >= 1:
                print(node)
            ts: ParserStructureStack = ParserStructureStack(node.structure)
            # Major optimization hack (24s -> 8s):
            # If the least common english letter matches, we try all possible strings.
            # If not, then we will only advance letter-by-letter, special words, or with non-alphabetic tokens.
            alpha_guess = node.n('z', ts)
            for token_id, token_string in tokens_to_process:
                if ((len(token_string) == 1 and ord(token_string[0]) <= 0xff) or
                        token_string in self.special_words or
                        not token_string.strip().isalpha() or
                        alpha_guess is not None):
                    token_char_list = token_char_lists[token_id]
                    key: tuple[ParserNode, LinkedChars] = (node, token_char_list)
                    tokens_calculated[(node.node_key, ll_to_str(token_char_list))] = set()
                    token_queue.append(key)

        idx: int = 0
        # This uses a stupid queue pattern, a list of all elements ever added to the queue, and an increasing index.
        while idx < len(token_queue):
            node = token_queue[idx][0]
            orig_token_char_list: LinkedChars = token_queue[idx][1]
            token_char_list = orig_token_char_list
            node_id: SingleNodeKey = node.node_key
            orig_node_key: NodeKeyTuple = (node_id,)
            idx += 1
            if node:
                root_structure: Structure = node.structure
                new_node: Optional[ParserNode] = node

                structure_stack = ParserStructureStack(node.structure)
                structure_stack.stack[:] = [node.structure]
                structure_stack.ops.clear()

                num_chars: int = 0
                assert (new_node is not None)  # To help mypy
                if not token_char_list:
                    new_node = root_structure.tick(structure_stack, new_node, '')
                else:
                    while token_char_list:
                        this_char: str
                        next_token_char_list: LinkedChars
                        this_char, next_token_char_list = token_char_list
                        new_node = root_structure.tick(structure_stack, new_node, this_char)
                        if type(new_node) is Structure:
                            break
                        if new_node is None:
                            # Illegal token
                            structure_stack.ops = []
                            if VERBOSE >= 3:
                                print(" -" + str(node_id) + "\"" + str(node)[:10] + "\" Got None at " +
                                      str(num_chars) + " of " + ll_to_str(orig_token_char_list))
                            break
                        num_chars += 1
                        token_char_list = next_token_char_list
                new_ops: OpTuple = ()
                for op in structure_stack.ops:
                    if type(op) is StructureNode:
                        assert len(op.node_key) == 3  # StructureNode node_key always has tuple (containing_struct, index, child_struct)
                        new_ops += (op.node_key,)
                    elif type(op) is str:
                        if new_node is not None:
                            new_ops += (op,)
                    elif op is None:
                        new_ops += (None,)
                if isinstance(new_node, Structure):
                    if VERBOSE >= 3:
                        print("  " + str(node_id) + "\"" + str(node)[:10] + "\" Got Structure " + str(new_node.name) +
                              " ops " + str(new_ops) + " at " + str(num_chars) + " of " + ll_to_str(orig_token_char_list))
                    token_substr: str = ll_to_str(token_char_list)
                    for parent, outer_token_node in entry_point_parents.get(new_node, []):
                        new_key_id = (parent.node_key, token_substr)
                        if new_key_id not in tokens_calculated:
                            tokens_calculated[new_key_id] = set()
                            token_queue.append((parent, token_char_list))
                        tokens_calculated[new_key_id].add((new_ops, outer_token_node, (node, orig_token_char_list)))
                else:
                    if new_node is not None:
                        assert (num_chars == len(ll_to_str(orig_token_char_list)))
                        if VERBOSE >= 2:
                            print("<>" + str(node_id) + "\"" + str(node)[:10] + "\" Got Node " + str(new_node.node_key)
                                  + " ops " + str(new_ops) + " at " + ll_to_str(orig_token_char_list))
                        if orig_node_key not in precalculated_raw_graph:
                            assert (len(orig_node_key[0]) > 1 and type(orig_node_key[0][1]) is int)
                            precalculated_raw_graph[orig_node_key] = {}
                        precalculated_raw_graph[orig_node_key][ll_to_str(orig_token_char_list)] = (new_ops, new_node.node_key)

        if VERBOSE >= 1:
            print("Done 1")
        for node_key_tuple, subdict in list(precalculated_raw_graph.items())[:]:
            for token_str, (new_ops, new_node_key) in list(subdict.items())[:]:
                this_key: tuple[SingleNodeKey, str] = (node_key_tuple[0], token_str)
                if this_key not in tokens_calculated:
                    continue
                recursion_queue = [(new_ops, tokens_calculated[this_key], node_key_tuple)]
                queue_idx: int = 0
                # This uses a stupid queue pattern, a list of all elements ever added to the queue, and an increasing index.
                while queue_idx < len(recursion_queue):
                    cur_ops: OpTuple = recursion_queue[queue_idx][0]
                    list_of_children: set[tuple[OpTuple, StructureNode, tuple[ParserNode, LinkedChars]]] = recursion_queue[queue_idx][1]
                    node_list: NodeKeyTuple = recursion_queue[queue_idx][2]
                    if len(node_list) > 100:
                        raise Exception("Infinite loop detected at " + str(recursion_queue[queue_idx]) + " | " + str((this_key, (new_ops, new_node_key))))
                    queue_idx += 1
                    for child_ops, outer_token_node, child_key in list_of_children:
                        child_inner_key: tuple[NodeKeyTuple, str] = (node_list[:-1] + (outer_token_node.node_key, child_key[0].node_key,), ll_to_str(child_key[1]))
                        total_ops: OpTuple = child_ops + (None,) + cur_ops
                        if child_inner_key[0] not in precalculated_raw_graph:
                            precalculated_raw_graph[child_inner_key[0]] = {}
                        assert (len(child_inner_key[0][0]) > 1 and type(child_inner_key[0][0][1]) is int)
                        precalculated_raw_graph[child_inner_key[0]][child_inner_key[1]] = (total_ops, new_node_key)
                        if VERBOSE >= 2:
                            print("^^" + str(child_inner_key[0]) + "\"" + str(child_key[0])[:10] + "\" Got Node " +
                                  str(child_inner_key[0]) + " ops " + str(total_ops) + " at " + child_inner_key[1])
                        calc_key: tuple[SingleNodeKey, str] = (child_key[0].node_key, child_inner_key[1])
                        if calc_key in tokens_calculated:
                            recursion_queue.append((total_ops, tokens_calculated[calc_key], child_inner_key[0]))

        if VERBOSE >= 2:
            print("Precalculated tokens:")
            for node_key, subdict in precalculated_raw_graph.items():
                for char_key, v in subdict.items():
                    print("    \"%s\" %s: %s" % (char_key, node_key, v))
        if VERBOSE >= 1:
            print("Done 2")
        return precalculated_raw_graph


class TokenizerData:
    """
    Holding object for the list of model strings and some default tensors.
    """

    def __init__(self, token_strings: list[str], eos_token_id: int = -1, model_device: str = 'cpu'):
        self.token_strings: list[str] = token_strings
        self.tokens_to_id: dict[str, int] = {}
        self.eos_token_id: int = eos_token_id
        for token_id, token_string in enumerate(self.token_strings):
            self.tokens_to_id[token_string] = token_id
        self.zero_tensor: Any = TokenTensor.create_tensor(self)
        self.model_device: str = model_device


class TokenSet(Iterable[int]):
    """
    Python set of token ids which can be used for intermediate calculations
    """

    def __init__(self, tokenizer_data: TokenizerData) -> None:
        self.child_sets: list[TokenSet | Iterable[int]] = []
        self.cow: bool = False
        self.tokenizer_data: TokenizerData = tokenizer_data
        self.token_strings: list[str] = tokenizer_data.token_strings
        self.token_set: set[int] = set()

    def __bool__(self) -> bool:
        return bool(self.token_set) or any(bool(x) for x in self.child_sets)

    def __contains__(self, item: int) -> bool:
        return item in self.token_set or any(item in x for x in self.child_sets)

    def __len__(self):
        self.flatten(True)
        return len(self.token_set)

    def __iter__(self):
        self.flatten(True)
        return iter(self.token_set)

    def init_tensor(self) -> None:
        pass

    def to_str_set(self) -> set[str]:
        self.flatten(True)
        return set(self.token_strings[tok] for tok in self.token_set)

    def add(self, tok: int):
        self.flatten(False)
        self.token_set.add(tok)

    def remove(self, tok: int):
        self.flatten(False)
        self.token_set.remove(tok)

    def update(self, other: TokenSet | Iterable[int]):
        if not self.token_set:
            if type(other) is TokenSet:
                other.flatten(True)
                self.token_set = other.token_set
                self.cow = True
            elif type(other) is set:
                self.token_set = other
                self.cow = True
            else:
                self.token_set = set(other)
        else:
            self.child_sets.append(other)

    def flatten(self, read: bool) -> None:
        if self.cow and (self.child_sets or not read):
            self.token_set = self.token_set.copy()
            self.cow = False
        if self.child_sets:
            for child in self.child_sets:
                self._update_internal(child)
            self.child_sets.clear()

    def _update_internal(self, other: TokenSet | Iterable[int]) -> None:
        if type(other) is TokenSet:
            cast(TokenSet, other).flatten(True)
            self.token_set.update(other.token_set)
        else:
            self.token_set.update(cast(Iterable[int], other))

    def intersection_update(self, other: TokenSet | Iterable[int]) -> None:
        self.flatten(False)
        if type(other) is TokenSet:
            cast(TokenSet, other).flatten(True)
            self.token_set.intersection_update(other.token_set)
        else:
            self.token_set.intersection_update(cast(Iterable[int], other))

    def difference_update(self, other: TokenSet | Iterable[int]) -> None:
        self.flatten(False)
        if type(other) is TokenSet:
            cast(TokenSet, other).flatten(True)
            self.token_set.difference_update(other.token_set)
        else:
            self.token_set.difference_update(cast(Iterable[int], other))

    @staticmethod
    def create_tensor(tokenizer_data: TokenizerData) -> Any:
        return None

    def to_model_device(self):
        pass


TokenTensor: TypeAlias = TokenSet

FALSE_MYPY_STUB: bool = False
if FALSE_MYPY_STUB:
    class torch:
        class BoolTensor:
            """
            Wrapper to assist mypy validation
            """
            @overload
            def __getitem__(self, idx: int) -> bool:
                ...

            @overload
            def __getitem__(self, idx: slice) -> 'torch.BoolTensor':
                ...

            def __getitem__(self, idx: int | slice) -> bool | 'torch.BoolTensor':
                return False if type(idx) is int else self

            def __or__(self, other: torch.BoolTensor) -> torch.BoolTensor:
                return self

            def __and__(self, other: torch.BoolTensor) -> torch.BoolTensor:
                return self

            def __invert__(self) -> torch.BoolTensor:
                return self

            def __setitem__(self, idx: int, val: bool):
                pass

            def item(self) -> bool:
                return False

            def to(self, device: str) -> torch.BoolTensor:
                return self

        @staticmethod
        def zeros(wid: int, dtype: Any = None) -> torch.BoolTensor:
            return torch.BoolTensor()

        @staticmethod
        def any(tensor: torch.BoolTensor) -> torch.BoolTensor:
            return tensor

        bool = bool


try:
    import torch  # type:ignore

    class TokenTensorImpl(TokenSet):
        """
        A tensor-optimized implementation of TokenSet, containing some efficient set operations.
        """
        def __init__(self, tokenizer_data: TokenizerData, token_set: None | TokenSet | Iterable[int] = None) -> None:
            super().__init__(tokenizer_data)
            self.tensor: torch.BoolTensor
            self.is_uninitialized: bool = False
            self.is_nonempty: bool = False
            if isinstance(token_set, TokenTensorImpl):
                self.tensor = token_set.tensor
                self.cow = True
            else:
                self.is_uninitialized = True
                if token_set:
                    self.init_tensor()
                    self.is_nonempty = True
                    for tok in token_set:
                        self.tensor[tok] = True

        @staticmethod
        def create_tensor(tokenizer_data: TokenizerData) -> torch.BoolTensor:
            return cast(torch.BoolTensor, torch.zeros(len(tokenizer_data.token_strings), dtype=torch.bool))

        def init_tensor(self) -> None:
            """
            Lazy-initializes this tensor (Tensors which have not been set will remain null)
            """
            if self.is_uninitialized:
                self.tensor = self.create_tensor(self.tokenizer_data)
                self.is_uninitialized = False

        def __bool__(self) -> bool:
            return not self.is_uninitialized and (self.is_nonempty or torch.any(self.tensor).item())

        def __contains__(self, item: int) -> bool:
            return not self.is_uninitialized and self.tensor[item]

        def __len__(self):
            return 0 if self.is_uninitialized else torch.dot(self.tensor, self.tensor)

        def __iter__(self):
            return (tok for tok in []) if self.is_uninitialized else (tok for tok in range(len(self.token_strings)) if self.tensor[tok])

        def to_str_set(self) -> set[str]:
            return set() if self.is_uninitialized else set(self.token_strings[tok] for tok in range(len(self.token_strings)) if self.tensor[tok])

        def add(self, tok: int):
            self.is_nonempty = True
            self.init_tensor()
            if self.cow:
                self.tensor = self.tensor[:]
            self.tensor[tok] = True

        def remove(self, tok: int):
            self.is_nonempty = False
            self.init_tensor()
            if self.cow:
                self.tensor = self.tensor[:]
            self.tensor[tok] = False

        def update(self, other: TokenSet | Iterable[int]):
            self._update_internal(other)

        def _update_internal(self, other: TokenSet | Iterable[int]):
            tt_tmp: TokenTensorImpl = TokenTensorImpl(self.tokenizer_data, other)
            if self.is_uninitialized:
                self.tensor = tt_tmp.tensor
                self.is_nonempty = tt_tmp.is_nonempty
                self.cow = True
            else:
                if self.cow:
                    self.tensor = self.tensor[:]
                self.tensor |= TokenTensorImpl(self.tokenizer_data, other).tensor
                self.is_nonempty = self.is_nonempty or tt_tmp.is_nonempty

        def intersection_update(self, other: TokenSet | Iterable[int]):
            self.is_nonempty = False
            if self.is_uninitialized:
                return
            if self.cow:
                self.tensor = self.tensor[:]
            self.tensor &= TokenTensorImpl(self.tokenizer_data, other).tensor

        def difference_update(self, other: TokenSet | Iterable[int]):
            self.is_nonempty = False
            if self.is_uninitialized:
                return
            if self.cow:
                self.tensor = self.tensor[:]
            self.tensor &= ~TokenTensorImpl(self.tokenizer_data, other).tensor

        def to_model_device(self):
            """
            Copies this tensor to the GPU
            """
            if self.tokenizer_data.model_device != 'cpu':
                self.tensor = self.tensor.to(self.tokenizer_data.model_device)

    TokenTensor = TokenTensorImpl  # type: ignore

    def get_tensor(token_set: TokenTensor) -> torch.BoolTensor:
        tt: TokenTensorImpl = cast(TokenTensorImpl, token_set)
        tt.init_tensor()
        return tt.tensor

except ImportError:
    torch = None  # type: ignore


def to_tensor(token_set: TokenSet) -> TokenTensor:
    if torch is not None:
        return TokenTensorImpl(token_set.tokenizer_data, token_set)
    else:
        return token_set


def get_zero_tensor(tokenizer_data: TokenizerData) -> Any:
    if torch is not None:
        return TokenTensorImpl.create_tensor(tokenizer_data)
    else:
        return set()


MAX_HISTORY_LENGTH = 4


class PrecalculatedStructureGraph:
    """
    Holds pre-calculated GPU tensors for every possible state (JSON has about 800 states, including nesting/recursion).
    """

    def __init__(self, precalculated_raw_graph: PrecalculatedRawGraph, tokenizer_data: TokenizerData, structure_graph: NodeStructureGraph) -> None:
        tokens_to_id: dict[str, int] = {}
        self.tokenizer_data = tokenizer_data
        self.token_strings: list[str] = tokenizer_data.token_strings
        for token_id, token_string in enumerate(self.token_strings):
            tokens_to_id[token_string] = token_id
        self.precalculated_raw_graph: PrecalculatedRawGraph = precalculated_raw_graph
        self.string_structures: list[str] = [s.name for s in structure_graph.structures.values() if s.is_string]
        self.recursive_structures: list[str] = [s.name for s in structure_graph.structures.values() if not s.is_string]
        self.root_node_key: SingleNodeKey = (structure_graph.root.name, 1)
        for key in self.precalculated_raw_graph.keys():
            if len(key) == 1 and key[0][:2] == self.root_node_key:
                self.root_node_key = key[0]
        self.special_tokens: set[int] = set(i for i, val in enumerate(self.token_strings) if val == '')
        self.space_tokens: dict[int, str] = dict((i, val) for i, val in enumerate(self.token_strings) if val and val.count(' ') == len(val))
        self.precalculated_vectors_by_state_key: dict[NodeKeyTuple, TokenTensor] = {}
        for state_key in self.precalculated_raw_graph.keys():
            ts: TokenSet = TokenSet(self.tokenizer_data)
            for i in range(1, len(state_key) + 1):
                possible_tokens: Iterable[str] = self.precalculated_raw_graph.get(state_key[-i:], [])
                for tok_str in possible_tokens:
                    if tok_str not in tokens_to_id:
                        # imaginary intermediate tokens will exist in precalculated_raw but must not be used.
                        continue
                    tok_id: int = tokens_to_id[tok_str]
                    if (state_key[-1][-1] == 'indent' or state_key[-1][-1] == 'indent_end') and tok_id in self.space_tokens:
                        continue  # indents are handled in a separate case
                    ts.add(tok_id)
            self.precalculated_vectors_by_state_key[state_key] = to_tensor(ts)
            self.precalculated_vectors_by_state_key[state_key].to_model_device()
        self.null_tensor: TokenTensor = to_tensor(TokenSet(self.tokenizer_data))
        self.null_tensor.init_tensor()
        self.null_tensor.to_model_device()
        self.history_length = MAX_HISTORY_LENGTH
        self.end_tensor: TokenTensor = to_tensor(TokenSet(self.tokenizer_data))
        if self.tokenizer_data.eos_token_id >= 0:
            self.end_tensor.add(self.tokenizer_data.eos_token_id)
        else:
            self.end_tensor.update(self.special_tokens)
        self.end_tensor.to_model_device()
        self.space_tensor: TokenTensor = to_tensor(TokenSet(self.tokenizer_data))
        self.space_tensor.update(self.space_tokens)
        self.space_tensor.to_model_device()
        self.space_tensors_by_length: list[TokenTensor] = []
        for num_spaces in range(1, 1 + max(len(sp) for sp in self.space_tokens.values())):
            tensor: TokenTensor = to_tensor(TokenSet(self.tokenizer_data))
            tensor.add(max((len(sp), idx) for idx, sp in self.space_tokens.items() if len(sp) <= num_spaces)[1])
            tensor.to_model_device()
            self.space_tensors_by_length.append(tensor)


class StructureExecutionEngine:
    """
    Main entry point: an executor which maintains a stack and state.
    Uses the pre-calculated tensors from PrecalculatedStructureGraph to quickly look up the next token mask in constant time.
    """

    def __init__(self, precalculated_structure_graph: PrecalculatedStructureGraph, indent_space_size: int) -> None:
        self.precalculated_structure_graph: PrecalculatedStructureGraph = precalculated_structure_graph
        self.precalculated_raw_graph: PrecalculatedRawGraph = precalculated_structure_graph.precalculated_raw_graph
        self.precalculated_vectors_by_state_key: dict[NodeKeyTuple, TokenTensor] = precalculated_structure_graph.precalculated_vectors_by_state_key
        self.string_structures: list[str] = precalculated_structure_graph.string_structures
        self.recursive_structures: list[str] = precalculated_structure_graph.recursive_structures
        self.root_node_key: SingleNodeKey = precalculated_structure_graph.root_node_key
        self.tokenizer_data: TokenizerData = precalculated_structure_graph.tokenizer_data
        self.token_strings: list[str] = precalculated_structure_graph.token_strings
        self.special_tokens: set[int] = precalculated_structure_graph.special_tokens
        self.space_tokens: dict[int, str] = precalculated_structure_graph.space_tokens
        self.end_tensor: TokenTensor = precalculated_structure_graph.end_tensor
        self.null_tensor: TokenTensor = precalculated_structure_graph.null_tensor
        self.space_tensor: TokenTensor = precalculated_structure_graph.space_tensor
        self.space_tensors_by_length: list[TokenTensor] = precalculated_structure_graph.space_tensors_by_length

        self.history_length = precalculated_structure_graph.history_length
        self.use_schema: bool = False
        self.reached_end: bool = False
        self.indent_space_size = indent_space_size
        self.indents_needed: int = 0
        self.struct_stack: list[SingleNodeKey] = []
        self.init()

    def init(self) -> None:
        """
        Reset the state machine to the beginning.
        """
        self.set_state(self.root_node_key)

    def set_state(self, *nodes: SingleNodeKey) -> None:
        """
        Reset the state machine to a specific state: may also be used to backtrack.
        """
        self.reached_end = False
        self.struct_stack = list(nodes)

    def get_logit_weights_tok(self) -> TokenTensor:
        """
        Returns a tensor representing the allowed tokens from the current state.
        """
        node_key: NodeKeyTuple = tuple(self.struct_stack[-self.history_length:])
        acceptable_token_set: TokenTensor
        sub_key: NodeKeyTuple = node_key
        while len(sub_key) > 1 and sub_key not in self.precalculated_vectors_by_state_key:
            sub_key = sub_key[1:]
        if sub_key in self.precalculated_vectors_by_state_key:
            acceptable_token_set = self.precalculated_vectors_by_state_key[sub_key]
        else:
            acceptable_token_set = self.end_tensor
            self.reached_end = True

        if self.struct_stack[-1][-1] == 'indent' or self.struct_stack[-1][-1] == 'indent_end':
            if self.indents_needed <= 0:
                # acceptable_token_set.difference_update(self.space_tensor)
                pass  # We already exclude space_tokens from indents in precalculated_vectors_by_state_key
            elif self.indents_needed < len(self.space_tensors_by_length):
                acceptable_token_set = self.space_tensors_by_length[self.indents_needed]
            else:
                acceptable_token_set = self.space_tensors_by_length[-1]
        # if not acceptable_token_set:
        #     acceptable_token_set = self.end_tensor
        #     self.reached_end = True
        return acceptable_token_set  # TokenSet(self.token_strings) # acceptable_token_set

    def get_logit_weights_str(self) -> set[str]:
        """
        Like `get_logit_weights_tok` but converts the resulting tokes to a human-readable form for debugging.
        """
        ret_tensor: TokenTensor = self.get_logit_weights_tok()
        possible_token_chars: set[str] = set()
        for token in ret_tensor:
            possible_token_chars.add(self.token_strings[token])
        return possible_token_chars

    def __call__(self, selected_token_str: Optional[str] = None) -> TokenTensor:
        """
        Helper method to combine `execute_str` and `get_logit_weights_tok`
        """
        if self.reached_end:
            return self.end_tensor
        if selected_token_str is not None:
            self.execute_str(selected_token_str)
        return self.get_logit_weights_tok()

    def n(self, selected_token_str: Optional[str] = None) -> set[str]:
        """
        Helper method to combine `execute_str` and `get_logit_weights_str`
        """
        if self.reached_end:
            return set()
        if selected_token_str is not None:
            self.execute_str(selected_token_str)
        return self.get_logit_weights_str()

    def get_precalculated_transition(self, selected_token_str: str) -> Optional[tuple[OpTuple, SingleNodeKey]]:
        """
        Returns the first matching tuple of operation list and new node, for the provided token and the current state.
        """
        node_key: NodeKeyTuple = tuple(self.struct_stack[-self.history_length:])
        for i in range(len(node_key)):
            tmp_key: NodeKeyTuple = node_key[-i - 1:]
            if tmp_key in self.precalculated_raw_graph and selected_token_str in self.precalculated_raw_graph[tmp_key]:
                return self.precalculated_raw_graph[tmp_key][selected_token_str]
        return None

    def execute_str(self, selected_token_str: str) -> SingleNodeKey:
        """
        Updates the internal state machine given the provided token string.
        """
        transition_data: Optional[tuple[OpTuple, SingleNodeKey]] = self.get_precalculated_transition(selected_token_str)
        if transition_data is None:
            # Unable to access the selected token from the current state. We have decoded an invalid state
            raise Exception("Failed to lookup token " + str(selected_token_str) + " from state " + str(self.struct_stack[-self.history_length:]))
        ops: OpTuple
        new_state: SingleNodeKey
        ops, new_state = transition_data

        prev_state = self.struct_stack.pop()
        for op in ops:
            if op is None:
                self.struct_stack.pop()
            elif type(op) is tuple:
                self.struct_stack.append(op)
        self.struct_stack.append(new_state)
        if new_state[-1] == 'indent' or new_state[-1] == 'indent_end':
            if prev_state != new_state:
                self.indents_needed = self.indent_space_size * (len(self.struct_stack) - (2 if new_state[-1] == 'indent_end' else 1))
            elif selected_token_str.strip() == '':
                self.indents_needed -= len(selected_token_str)
            else:
                raise Exception("Invalid indentation " + str(selected_token_str) + " at state " + str(self.struct_stack) + ": " + str(self.indents_needed))
        return new_state


class JsonNodeStructureGraph(NodeStructureGraph):
    """
    A JSON grammar, with a small tweak to make it more friendly for text generation: prohibiting empty objects.
    """
    def __init__(self) -> None:
        super().__init__()
        self.number: Structure = Structure("number").init(self)
        self.string: Structure = Structure("string").init(self)
        self.array: Structure = Structure("array").init(self)
        self.json_obj: Structure = Structure("object").init(self)
        self.constant: Structure = Structure("constant").init(self)
        # self.json: Structure = Structure("json").init(self)

        value = (self.string, self.constant, self.json_obj, self.array, self.number)

        onenine = trange('1', '9')
        digit = trange('0', '9')
        fraction = ['.', digit, ZeroOrMore(digit)]
        integer = [OptionalNode('-'), ('0', [onenine, ZeroOrMore(digit)])]
        exponent_sign = ('+', '-')
        exponent = [('E', 'e'), ([exponent_sign, digit], digit), ZeroOrMore(digit)]
        hex_digit = RegexRangeNode('[0-9a-fA-F]')  # Equivalent to (trange('0', '9'), trange('a', 'f'), trange('A', 'F'))
        escape = (('"', '\\', '/', 'b', 'f', 'n', 'r', 't'), ['u', hex_digit, hex_digit, hex_digit, hex_digit])
        character = (RegexRangeNode('[^\u0001-\u001f\"\\]'), ['\\', escape])

        element = value
        elements = [element, ZeroOrMore([',', OptionalNode((' ', ['\n', AutoIndent(' ')])), element])]
        member = [self.string, ':', OptionalNode(' '), element]
        members = [member, ZeroOrMore([',', OptionalNode((' ', ['\n', AutoIndent(' ')])), member])]

        self.init_structure(self.string, ['"', ZeroOrMore(character), '"'])
        self.init_structure(self.number, [integer, OptionalNode(fraction), OptionalNode(exponent)])
        # self.init_structure(self.array, ['[', (']', elements + [OptionalNode(['\n', AutoIndent(' ')]), ']'])])
        # self.init_structure(self.json_obj, ['{', ('}', members + [OptionalNode(['\n', AutoIndent(' ')]), '}'])])

        # Forbid empty objects:
        self.init_structure(self.array, ['[', OptionalNode(['\n', AutoIndent(' ')]), (elements + [OptionalNode(['\n', AutoIndentEnd(' ')]), ']'])])
        self.init_structure(self.json_obj, ['{', OptionalNode(['\n', AutoIndent(' ')]), (members + [OptionalNode(['\n', AutoIndentEnd(' ')]), '}'])])

        self.init_structure(self.constant, ("true", "false", "null"))
        # self.init_structure(self.json, (self.json_obj, self.array))

        self.init_graph()
        self.root = self.json_obj


# Testing code
TEST_TOKENS: list[str] = [
    '</s>', '\n', '!', '"', '""', '")', '"))', '"));', '"),', '").', '");', '")]', '")`',
    '"+', '",', '","', '".', '".$', '"/', '"/>', '":', '":"', '":{"', '";', '"=>', '">', '"><',
    '"></', '"?', '"?>', '"\\', '"]', '"])', '"],', '"].', '"];', '"`', '"}', '"},', '#', '$',
    '%', '&', '\'', '(', ')', '*', '+', ',', ', ', ',\n', ',"', ',-', ',[', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
    'A', '', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '["', '[-', '[[', '[]', '[{', '\\',
    ']', '],', '],[', ']]', '^', '_', '`', 'a', '', 'c', 'd', 'e', 'f', 'false', 'g', 'h',
    'i', 'j', 'k', 'l', 'm', 'n', 'null', 'o', 'p', 'q', 'r', 's', 't', 'true', 'u', 'v',
    'w', 'x', 'y', 'z', '{', '{"', '{{', '{}', '|', '}', '}}', '~', ' ', '  ', '   ',  # '\u2581\u2581\u2581'
]


def cleanup_string_test_tokens(token_strings: list[str]) -> list[str]:
    token_strings = token_strings[:]
    for i in range(len(token_strings)):
        s: str = token_strings[i].replace("\r", "\n")
        if s == '<s>' or s == '</s>':
            s = ''
        if s.startswith("<0x") and s.endswith('>'):
            s = chr(int(s[1:-1], 16))
        token_strings[i] = s.replace('\u2581', ' ')
    return token_strings


def run_tests() -> StructureExecutionEngine:
    test_tokens: list[str] = cleanup_string_test_tokens(TEST_TOKENS)

    test_graph: NodeStructureGraph = JsonNodeStructureGraph()
    test_precalculated_raw: PrecalculatedRawGraph = test_graph.calculate_structure_graph(test_tokens)
    test_precalculated_graph: PrecalculatedStructureGraph = PrecalculatedStructureGraph(test_precalculated_raw, TokenizerData(test_tokens), test_graph)
    test_engine: StructureExecutionEngine = StructureExecutionEngine(test_precalculated_graph, 0)
    test_engine.init()
    try:
        # Start with an array
        test_engine.n('{"')
        test_engine.n('":')
        assert sorted(test_engine.n('[')) == [
            '\n', '"', '""', '")', '"))', '"));', '"),', '").', '");', '")]', '")`', '"+', '",', '","', '".', '".$',
            '"/', '"/>', '":', '":"', '":{"', '";', '"=>', '">', '"><', '"></', '"?', '"?>', '"\\', '"]', '"])',
            '"],', '"].', '"];', '"`', '"}', '"},', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '[', '["', '[-', '[[', '[{', 'f', 'false', 'n', 'null', 't', 'true', '{', '{"']
        # '[', '["', '[-', '[[', '[]', '[{', ']', 'f', 'false', 'n', 'null', 't', 'true', '{', '{"', '{}']

        # Test a string, for example starting with some symbols
        assert len(test_engine.n('")]')) > 100  # Contains all words and letters
        assert len(test_engine.n('n')) > 100  # Contains all words and letters
        escape_chars = [
            '"', '""', '")', '"))', '"));', '"),', '").', '");', '")]', '")`', '"+', '",', '","', '".', '".$',
            '"/', '"/>', '":', '":"', '":{"', '";', '"=>', '">', '"><', '"></', '"?', '"?>', '"\\', '"]', '"])',
            '"],', '"].', '"];', '"`', '"}', '"},', '/', '\\', 'f', 'false', 'n', 'null', 'r', 't', 'true', 'u']
        assert sorted(test_engine.n('\\')) == escape_chars
        assert sorted(test_engine.n('"\\')) == escape_chars
        hex_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'C', 'D', 'E', 'F', 'a', 'c', 'd', 'e', 'f']
        assert sorted(test_engine.n('u')) == hex_chars
        assert sorted(test_engine.n('a')) == hex_chars
        assert len(list(test_engine.n())) == len(list(test_engine()))
        # The 'false' thing is a quirk but the first chars are hex, and we allow "special words" anywhere in the graph
        assert sorted(test_engine.n('3')) == hex_chars + ['false']
        assert sorted(test_engine.n('2')) == hex_chars + ['false']
        assert len(list(test_engine('2'))) > 100  # Done with unicode escape.
        assert sorted(test_engine.n('",')) == [
            '\n', ' ', '"', '""', '")', '"))', '"));', '"),', '").', '");', '")]', '")`', '"+', '",', '","', '".',
            '".$', '"/', '"/>', '":', '":"', '":{"', '";', '"=>', '">', '"><', '"></', '"?', '"?>', '"\\',
            '"]', '"])', '"],', '"].', '"];', '"`', '"}', '"},', '-', '0', '1', '2', '3', '4', '5', '6', '7',
            '8', '9', '[', '["', '[-', '[[', '[{', 'f', 'false', 'n', 'null', 't', 'true', '{', '{"']
        # '8', '9', '[', '["', '[-', '[[', '[]', '[{', ']', 'f', 'false', 'n', 'null', 't', 'true', '{', '{"', '{}']
        # Now, test a number
        assert sorted(test_engine.n('[-')) == ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        assert sorted(test_engine.n('0')) == ['\n', ',', ',\n', ', ', ',"', ',-', ',[', '.', 'E', ']', '],', '],[', ']]', 'e']
        # in the middle of a number, ",-" will exit the number and enter a new number, and append a '-' character to that number.
        # assert eng.precalculated_raw_graph[tuple(eng.struct_stack)[-2:]][',-'] == ((None, ('array', 25, 'number'), '-'), ('number', 5))
        tran1 = cast(Tuple[Tuple[None, StructureNodeKey, str], SingleNodeKey], test_engine.get_precalculated_transition(',-'))
        assert tran1 == ((None, ('array', 0 + tran1[0][1][1], 'number'), '-'), ('number', 5))
        tran2 = cast(Tuple[Tuple[None, None, StructureNodeKey], SingleNodeKey], test_engine.get_precalculated_transition('],['))
        assert test_engine.get_precalculated_transition('],[') == ((None, None, ('array', 0 + tran2[0][2][1], 'array')), ('array', 3))
        assert sorted(test_engine.n('E')) == ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        assert test_engine.get_precalculated_transition(',') is None
        assert sorted(test_engine.n('+')) == ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        assert sorted(test_engine.n('1')) == [
            '\n', ',', ',\n', ', ', ',"', ',-', ',[', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ']', '],', '],[', ']]']
        test_engine.n(']]')
        assert test_engine.n('}') == {''}
        assert test_engine.n('') == set()
        test_engine.init()
        # assert test_engine.n('{}') == {''}
        # assert test_engine.n('') == set()
        # test_engine.init()
        print("Test successful")
    except Exception:
        print(sorted(test_engine.n()))
        import traceback
        traceback.print_exc()
    return test_engine


def run_perf_tests() -> StructureExecutionEngine:
    import time
    test_token_set: set[str] = set(TEST_TOKENS)
    tmpl: list[str] = list(test_token_set)[:]
    for tok1 in tmpl:
        for tok2 in tmpl:
            # and '\n' not in tok1 and '\n' not in tok2 and '\u2581' not in tok1 and '\u2581' not in tok2
            if not tok1.isalnum() and not tok2.isalnum():
                test_token_set.add(tok1 + tok2)
    for a in 'abcdefghijklmnopqrstuvwxyz':
        for b in 'abcdefghijklmnopqrstuvwxyz':
            ab = a + b
            test_token_set.add(ab)
            for c in 'abcdefghijklmnopqrstuvwxyz':
                test_token_set.add(ab + c)
    test_tokens: list[str] = sorted(list(test_token_set))
    test_tokens = cleanup_string_test_tokens(test_tokens)
    t1: float = time.time()
    test_graph: NodeStructureGraph = JsonNodeStructureGraph()
    test_precalculated_raw: PrecalculatedRawGraph = test_graph.calculate_structure_graph(test_tokens)
    t2: float = time.time()
    print("Precalculated raw graph took " + str(t2 - t1))
    test_precalculated_graph: PrecalculatedStructureGraph = PrecalculatedStructureGraph(test_precalculated_raw, TokenizerData(test_tokens), test_graph)
    t3: float = time.time()
    print("Precalculated structure graph took " + str(t3 - t2))
    test_engine: StructureExecutionEngine = StructureExecutionEngine(test_precalculated_graph, 2)
    test_engine.init()
    try:
        tokens_by_length = sorted(enumerate(test_tokens), key=lambda k: -len(k[1]))
        example_json = (
            "{\n   \"spaceShuttles\": [\n     {\n       \"name\": \"Columbia\",\n       \"launchDate\": \"April 12, 1981\"\n       },\n     {\n       \"name\": \"Challenger\"," +
            "\n       \"launchDate\": \"January 28, 1986\"\n       },\n     {\n       \"name\": \"Discovery\",\n       \"launchDate\": \"April 4, 1990\"\n       },\n     {" +
            "\n       \"name\": \"Endeavour\",\n       \"launchDate\": \"May 24, 1998\"\n       },\n     {\n       \"name\": \"Voyager 1\",\n       \"launchDate\": \"September 5, 1977\"" +
            "\n       },\n     {\n       \"name\": \"Voyager 2\",\n       \"launchDate\": \"September 22, 1977\"\n       }\n     ]\n"
        )
        token_ids: list[int] = []
        i = 0
        while i < len(example_json):
            token_ids.append(next(idx for (idx, s) in tokens_by_length if example_json[i: i + len(s)] == s))
            i += len(test_tokens[token_ids[-1]])
        print(len(token_ids))
        t5: float = time.time()
        test_count = 100
        for i in range(test_count):
            # print("Test #" + str(i))
            test_engine.init()
            for tid in token_ids:
                # print("Running token " + str(tid) + ": '" + str(test_tokens[tid]) + "'")
                test_engine(test_tokens[tid])
        t6: float = time.time()
        print("Test successful: " + str((t6 - t5)/test_count) + " runtime overheaad per " + str(len(token_ids)) + " tokens")
    except Exception:
        print(sorted(test_engine.n()))
        import traceback
        traceback.print_exc()
    return test_engine


if __name__ == '__main__':
    eng_test: StructureExecutionEngine
    if len(sys.argv) == 2 and sys.argv[1] in ('--test', 'test'):
        eng_test = run_tests()
    elif len(sys.argv) == 2 and sys.argv[1] in ('--perf', 'perf'):
        eng_test = run_perf_tests()
    else:
        raw_tokens: list[bytes]
        if len(sys.argv) > 1:
            # test_token: bytes = b'[3.14,[]]'
            # test_token: bytes = b'.14e+5'
            test_token: bytes = sys.argv[1].encode('utf-8')
            raw_tokens = [b'', b' ', test_token]
            print(raw_tokens)
        else:
            # To run tests, make a file tokens.txt with all model tokens separated by newlines.
            raw_tokens = open('tokens.txt', 'rb').read().split(b'\n')

        # \u2581 is the metaspace character: it is permitted if space is permitted.
        main_tokens: list[str] = ["" if s in [b'</s>', b'<s>'] else s.decode("utf-8", "replace") for s in raw_tokens]
        main_tokens = cleanup_string_test_tokens(main_tokens)
        main_tokenizer_data: TokenizerData = TokenizerData(main_tokens)
        graph: NodeStructureGraph = JsonNodeStructureGraph()
        precalculated_raw: PrecalculatedRawGraph = graph.calculate_structure_graph(main_tokens)
        precalculated_graph: PrecalculatedStructureGraph = PrecalculatedStructureGraph(precalculated_raw, main_tokenizer_data, graph)
        engine: StructureExecutionEngine = StructureExecutionEngine(precalculated_graph, 2)

        eng_test = run_tests()
