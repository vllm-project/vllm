// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! A set of small non-negative integers that iterates in exactly the order a
//! CPython (3.12) `set[int]` does.
//!
//! The encoder cache manager frees a request's cached inputs by iterating a
//! `set[int]` of input ids, and the order decides which entries become
//! evictable first. CPython iterates a set in hash-table slot order, which
//! for `hash(i) == i` is ascending only while every element is below the
//! table size, e.g. `{2, 9}` iterates as `[9, 2]`. This reproduces
//! `Objects/setobject.c`: open addressing with linear probing then
//! perturbation, dummy (deleted) slots reused by the next insertion probing
//! past them, and resizing once three fifths of the table is filled.

const MINSIZE: usize = 8;
const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Unused,
    Dummy,
    Active(usize),
}

#[derive(Clone, Debug)]
pub(crate) struct PyIntSet {
    table: Vec<Slot>,
    /// Active plus dummy slots.
    fill: usize,
    /// Active slots.
    used: usize,
}

impl Default for PyIntSet {
    fn default() -> Self {
        Self {
            table: vec![Slot::Unused; MINSIZE],
            fill: 0,
            used: 0,
        }
    }
}

impl PyIntSet {
    fn mask(&self) -> usize {
        self.table.len() - 1
    }

    /// The probe sequence of `set_lookkey` / `set_add_entry`, yielding slot
    /// indices until `f` returns `Some`.
    fn probe<R>(&self, key: usize, mut f: impl FnMut(usize, Slot) -> Option<R>) -> R {
        let mask = self.mask();
        let mut perturb = key;
        let mut i = key & mask;
        loop {
            let probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            for j in i..=i + probes {
                if let Some(r) = f(j, self.table[j]) {
                    return r;
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }

    /// The slot holding `key`, if present.
    fn find(&self, key: usize) -> Option<usize> {
        self.probe(key, |j, slot| match slot {
            Slot::Unused => Some(None),
            Slot::Active(k) if k == key => Some(Some(j)),
            _ => None,
        })
    }

    pub(crate) fn contains(&self, key: usize) -> bool {
        self.find(key).is_some()
    }

    pub(crate) fn add(&mut self, key: usize) {
        // `set_add_entry`: probe up to an unused slot, remembering the last
        // dummy slot on the way, which is reused if there is one (observed
        // on CPython 3.12; it is not the first one).
        let mut freeslot = None;
        let found = self.probe(key, |j, slot| match slot {
            Slot::Unused => Some(Some(j)),
            Slot::Active(k) if k == key => Some(None),
            Slot::Dummy => {
                freeslot = Some(j);
                None
            }
            Slot::Active(_) => None,
        });
        let Some(j) = found else {
            return;
        };
        self.used += 1;
        if let Some(freeslot) = freeslot {
            self.table[freeslot] = Slot::Active(key);
            return;
        }
        self.table[j] = Slot::Active(key);
        self.fill += 1;
        if self.fill * 5 >= self.mask() * 3 {
            let minused = if self.used > 50000 {
                self.used * 2
            } else {
                self.used * 4
            };
            self.resize(minused);
        }
    }

    /// Remove `key`, returning whether it was present (`set.discard`).
    pub(crate) fn discard(&mut self, key: usize) -> bool {
        let Some(j) = self.find(key) else {
            return false;
        };
        self.table[j] = Slot::Dummy;
        self.used -= 1;
        true
    }

    fn resize(&mut self, minused: usize) {
        let mut newsize = MINSIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        let old = std::mem::replace(&mut self.table, vec![Slot::Unused; newsize]);
        self.fill = self.used;
        for slot in old {
            if let Slot::Active(key) = slot {
                // `set_insert_clean`: the first unused slot on the probe path.
                let j = self.probe(key, |j, slot| (slot == Slot::Unused).then_some(j));
                self.table[j] = Slot::Active(key);
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.used == 0
    }

    /// The elements in CPython's iteration order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.table.iter().filter_map(|slot| match slot {
            Slot::Active(k) => Some(*k),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Deserialize)]
    struct Case {
        ops: Vec<Op>,
    }

    #[derive(Deserialize)]
    struct Op {
        op: String,
        value: usize,
        state: Vec<usize>,
    }

    /// Replays random add / discard sequences recorded from CPython and
    /// compares the iteration order after each step.
    #[test]
    fn matches_cpython_iteration_order() {
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("../../tests/fixtures/py_int_set_cases.json"))
                .unwrap();
        for (case_idx, case) in cases.iter().enumerate() {
            let mut set = PyIntSet::default();
            for (op_idx, op) in case.ops.iter().enumerate() {
                match op.op.as_str() {
                    "add" => set.add(op.value),
                    "discard" => {
                        set.discard(op.value);
                    }
                    other => panic!("unknown op {other}"),
                }
                let state: Vec<usize> = set.iter().collect();
                assert_eq!(state, op.state, "case {case_idx} op {op_idx}");
                assert_eq!(set.used, op.state.len());
                assert!(op.state.iter().all(|&k| set.contains(k)));
            }
        }
    }

    #[test]
    fn slot_order_is_not_sorted_order() {
        // `list({2, 9})` is `[9, 2]` in CPython.
        let mut set = PyIntSet::default();
        set.add(2);
        set.add(9);
        assert_eq!(set.iter().collect::<Vec<_>>(), [9, 2]);
        assert!(set.discard(9));
        assert!(!set.discard(9));
        assert!(!set.is_empty());
    }
}
