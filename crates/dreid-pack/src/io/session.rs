use crate::model::system::{MAX_SIDECHAIN_ATOMS, System};
use arrayvec::ArrayVec;

pub use dreid_forge::{BondOrder, Element, ResidueCategory, ResiduePosition, StandardResidue};

/// A packing session: the molecular system together with its topology and metadata.
#[derive(Debug, Clone)]
pub struct Session {
    /// Packable system (mobile residues + fixed scaffold).
    pub system: System,
    /// Topology + metadata bookkeeping for biological system.
    metadata: SystemMetadata,
}

impl Session {
    /// Creates a session from a `system` and its paired `metadata`.
    pub(super) fn new(system: System, metadata: SystemMetadata) -> Self {
        debug_assert_eq!(
            system.mobile.len(),
            metadata.mobile_residues.len(),
            "mobile residue count mismatch: system has {} but metadata has {}",
            system.mobile.len(),
            metadata.mobile_residues.len(),
        );
        debug_assert_eq!(
            system.fixed.positions.len(),
            metadata.fixed_atoms.len(),
            "fixed atom count mismatch: system has {} but metadata has {}",
            system.fixed.positions.len(),
            metadata.fixed_atoms.len(),
        );
        Self { system, metadata }
    }

    /// Topology metadata paired with this session.
    pub(super) fn metadata(&self) -> &SystemMetadata {
        &self.metadata
    }

    /// Residue locators for all mobile sidechains.
    ///
    /// Yields `(chain_id, residue_id, insertion_code)` for each packable
    /// residue in declaration order.
    pub fn mobile_residues(&self) -> impl Iterator<Item = (&str, i32, Option<char>)> + '_ {
        self.metadata
            .mobile_residues
            .iter()
            .map(|m| (m.chain_id.as_str(), m.residue_id, m.insertion_code))
    }
}

/// Topology bookkeeping that accompanies a [`System`].
#[derive(Debug, Clone)]
pub struct SystemMetadata {
    /// Periodic box vectors (Å), or `None`.
    pub box_vectors: Option<[[f64; 3]; 3]>,
    /// All covalent bonds.
    pub bonds: Vec<Bond>,
    /// One entry per atom in `System::fixed`, preserving order.
    pub fixed_atoms: Vec<FixedAtom>,
    /// One entry per residue in `System::mobile`, preserving order.
    pub mobile_residues: Vec<MobileSidechain>,
}

/// A covalent bond between two atoms.
#[derive(Debug, Clone)]
pub struct Bond {
    /// First endpoint.
    pub a: AtomRef,
    /// Second endpoint.
    pub b: AtomRef,
    /// Bond order.
    pub order: BondOrder,
}

/// Reference to an atom in either the fixed pool or a mobile sidechain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomRef {
    /// Index into fixed atom pool.
    Fixed(u32),
    /// Residue index + local atom index into mobile sidechain.
    Mobile { residue: u32, local: u8 },
}

/// Topology metadata for one atom in the fixed pool.
#[derive(Debug, Clone)]
pub struct FixedAtom {
    /// Atom name (e.g. `"CA"`, `"OD1"`).
    pub atom_name: String,
    /// Residue name (e.g. `"SER"`, `"HOH"`).
    pub residue_name: String,
    /// Residue sequence number.
    pub residue_id: i32,
    /// Chain identifier.
    pub chain_id: String,
    /// Insertion code, or `None` if absent.
    pub insertion_code: Option<char>,
    /// Matched standard residue name, or `None` for non-standard/HETATM.
    pub standard_name: Option<StandardResidue>,
    /// Residue category (standard, hetero, ion).
    pub category: ResidueCategory,
    /// Residue chain position (N-terminal, internal, C-terminal, …).
    pub position: ResiduePosition,
    /// Chemical element.
    pub element: Element,
}

/// Topology metadata for one mobile residue.
#[derive(Debug, Clone)]
pub struct MobileSidechain {
    /// Residue name.
    pub residue_name: String,
    /// Residue sequence number.
    pub residue_id: i32,
    /// Chain identifier.
    pub chain_id: String,
    /// Insertion code, or `None` if absent.
    pub insertion_code: Option<char>,
    /// Matched standard residue name, or `None` for non-standard residues.
    pub standard_name: Option<StandardResidue>,
    /// Residue category.
    pub category: ResidueCategory,
    /// Residue chain position.
    pub position: ResiduePosition,
    /// Sidechain atom elements, per local atom.
    pub elements: ArrayVec<Element, MAX_SIDECHAIN_ATOMS>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        system::{FixedAtomPool, ForceFieldParams, HBondParams, LjMatrix, VdwMatrix},
        types::{TypeIdx, Vec3},
    };
    use std::collections::{HashMap, HashSet};

    fn empty_system() -> System {
        System {
            mobile: vec![],
            fixed: FixedAtomPool {
                positions: vec![],
                types: vec![],
                charges: vec![],
                donor_for_h: vec![],
            },
            ff: ForceFieldParams {
                vdw: VdwMatrix::LennardJones(LjMatrix::new(0, vec![])),
                hbond: HBondParams::new(HashSet::new(), HashSet::new(), HashMap::new()),
            },
        }
    }

    fn system_with_n_fixed(n: usize) -> System {
        System {
            mobile: vec![],
            fixed: FixedAtomPool {
                positions: vec![Vec3::new(0.0, 0.0, 0.0); n],
                types: vec![TypeIdx(0); n],
                charges: vec![0.0f32; n],
                donor_for_h: vec![u32::MAX; n],
            },
            ff: ForceFieldParams {
                vdw: VdwMatrix::LennardJones(LjMatrix::new(0, vec![])),
                hbond: HBondParams::new(HashSet::new(), HashSet::new(), HashMap::new()),
            },
        }
    }

    fn empty_metadata() -> SystemMetadata {
        SystemMetadata {
            box_vectors: None,
            bonds: vec![],
            fixed_atoms: vec![],
            mobile_residues: vec![],
        }
    }

    fn fixed_atom() -> FixedAtom {
        FixedAtom {
            atom_name: "CA".to_string(),
            residue_name: "SER".to_string(),
            residue_id: 1,
            chain_id: "A".to_string(),
            insertion_code: None,
            standard_name: Some(StandardResidue::SER),
            category: ResidueCategory::Standard,
            position: ResiduePosition::Internal,
            element: Element::C,
        }
    }

    fn mobile_sidechain() -> MobileSidechain {
        MobileSidechain {
            residue_name: "SER".to_string(),
            residue_id: 1,
            chain_id: "A".to_string(),
            insertion_code: None,
            standard_name: Some(StandardResidue::SER),
            category: ResidueCategory::Standard,
            position: ResiduePosition::Internal,
            elements: ArrayVec::new(),
        }
    }

    #[test]
    fn atom_ref_variants_are_distinct() {
        assert_ne!(
            AtomRef::Fixed(0),
            AtomRef::Mobile {
                residue: 0,
                local: 0
            }
        );
    }

    #[test]
    fn atom_ref_is_copy() {
        let a = AtomRef::Fixed(7);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn atom_ref_fixed_roundtrips() {
        let AtomRef::Fixed(idx) = AtomRef::Fixed(42) else {
            panic!("unexpected variant");
        };
        assert_eq!(idx, 42);
    }

    #[test]
    fn atom_ref_mobile_roundtrips() {
        let AtomRef::Mobile { residue, local } = (AtomRef::Mobile {
            residue: 5,
            local: 3,
        }) else {
            panic!("unexpected variant");
        };
        assert_eq!(residue, 5);
        assert_eq!(local, 3);
    }

    #[test]
    fn atom_ref_fixed_max_index() {
        let AtomRef::Fixed(idx) = AtomRef::Fixed(u32::MAX) else {
            panic!("unexpected variant");
        };
        assert_eq!(idx, u32::MAX);
    }

    #[test]
    fn atom_ref_mobile_max_indices() {
        let AtomRef::Mobile { residue, local } = (AtomRef::Mobile {
            residue: u32::MAX,
            local: u8::MAX,
        }) else {
            panic!("unexpected variant");
        };
        assert_eq!(residue, u32::MAX);
        assert_eq!(local, u8::MAX);
    }

    #[test]
    fn atom_ref_fixed_eq_same_index() {
        assert_eq!(AtomRef::Fixed(0), AtomRef::Fixed(0));
        assert_ne!(AtomRef::Fixed(0), AtomRef::Fixed(1));
    }

    #[test]
    fn atom_ref_mobile_eq_same_pair() {
        assert_eq!(
            AtomRef::Mobile {
                residue: 2,
                local: 1
            },
            AtomRef::Mobile {
                residue: 2,
                local: 1
            },
        );
        assert_ne!(
            AtomRef::Mobile {
                residue: 2,
                local: 1
            },
            AtomRef::Mobile {
                residue: 2,
                local: 2
            },
        );
        assert_ne!(
            AtomRef::Mobile {
                residue: 2,
                local: 1
            },
            AtomRef::Mobile {
                residue: 3,
                local: 1
            },
        );
    }

    #[test]
    fn bond_stores_all_fields() {
        let b = Bond {
            a: AtomRef::Fixed(0),
            b: AtomRef::Mobile {
                residue: 1,
                local: 2,
            },
            order: BondOrder::Single,
        };
        assert_eq!(b.a, AtomRef::Fixed(0));
        assert_eq!(
            b.b,
            AtomRef::Mobile {
                residue: 1,
                local: 2
            }
        );
        assert_eq!(b.order, BondOrder::Single);
    }

    #[test]
    fn bond_all_orders_roundtrip() {
        for order in [
            BondOrder::Single,
            BondOrder::Double,
            BondOrder::Triple,
            BondOrder::Aromatic,
        ] {
            let b = Bond {
                a: AtomRef::Fixed(0),
                b: AtomRef::Fixed(1),
                order,
            };
            assert_eq!(b.order, order);
        }
    }

    #[test]
    fn bond_clone_is_equal() {
        let b = Bond {
            a: AtomRef::Fixed(3),
            b: AtomRef::Fixed(7),
            order: BondOrder::Double,
        };
        let c = b.clone();
        assert_eq!(c.a, b.a);
        assert_eq!(c.b, b.b);
        assert_eq!(c.order, b.order);
    }

    #[test]
    fn fixed_atom_stores_all_fields() {
        let fa = FixedAtom {
            atom_name: "OD1".to_string(),
            residue_name: "ASP".to_string(),
            residue_id: -3,
            chain_id: "B".to_string(),
            insertion_code: Some('A'),
            standard_name: Some(StandardResidue::ASP),
            category: ResidueCategory::Standard,
            position: ResiduePosition::CTerminal,
            element: Element::O,
        };
        assert_eq!(fa.atom_name, "OD1");
        assert_eq!(fa.residue_name, "ASP");
        assert_eq!(fa.residue_id, -3);
        assert_eq!(fa.chain_id, "B");
        assert_eq!(fa.insertion_code, Some('A'));
        assert_eq!(fa.standard_name, Some(StandardResidue::ASP));
        assert_eq!(fa.category, ResidueCategory::Standard);
        assert_eq!(fa.position, ResiduePosition::CTerminal);
        assert_eq!(fa.element, Element::O);
    }

    #[test]
    fn fixed_atom_optional_fields_none() {
        let fa = FixedAtom {
            atom_name: "O".to_string(),
            residue_name: "HOH".to_string(),
            residue_id: 999,
            chain_id: " ".to_string(),
            insertion_code: None,
            standard_name: None,
            category: ResidueCategory::Hetero,
            position: ResiduePosition::None,
            element: Element::O,
        };
        assert!(fa.insertion_code.is_none());
        assert!(fa.standard_name.is_none());
    }

    #[test]
    fn fixed_atom_clone_is_equal() {
        let fa = fixed_atom();
        let cl = fa.clone();
        assert_eq!(cl.atom_name, fa.atom_name);
        assert_eq!(cl.residue_id, fa.residue_id);
        assert_eq!(cl.element, fa.element);
        assert_eq!(cl.standard_name, fa.standard_name);
    }

    #[test]
    fn mobile_sidechain_stores_all_fields() {
        let ms = MobileSidechain {
            residue_name: "TRP".to_string(),
            residue_id: 100,
            chain_id: "C".to_string(),
            insertion_code: Some('B'),
            standard_name: Some(StandardResidue::TRP),
            category: ResidueCategory::Standard,
            position: ResiduePosition::NTerminal,
            elements: ArrayVec::new(),
        };
        assert_eq!(ms.residue_name, "TRP");
        assert_eq!(ms.residue_id, 100);
        assert_eq!(ms.chain_id, "C");
        assert_eq!(ms.insertion_code, Some('B'));
        assert_eq!(ms.standard_name, Some(StandardResidue::TRP));
        assert_eq!(ms.category, ResidueCategory::Standard);
        assert_eq!(ms.position, ResiduePosition::NTerminal);
    }

    #[test]
    fn mobile_sidechain_optional_fields_none() {
        let ms = MobileSidechain {
            residue_name: "UNK".to_string(),
            residue_id: 0,
            chain_id: "A".to_string(),
            insertion_code: None,
            standard_name: None,
            category: ResidueCategory::Hetero,
            position: ResiduePosition::None,
            elements: ArrayVec::new(),
        };
        assert!(ms.insertion_code.is_none());
        assert!(ms.standard_name.is_none());
    }

    #[test]
    fn mobile_sidechain_elements_empty_on_construction() {
        assert!(mobile_sidechain().elements.is_empty());
    }

    #[test]
    fn mobile_sidechain_elements_fills_to_capacity() {
        let mut ms = mobile_sidechain();
        for _ in 0..MAX_SIDECHAIN_ATOMS {
            ms.elements.push(Element::C);
        }
        assert_eq!(ms.elements.len(), MAX_SIDECHAIN_ATOMS);
    }

    #[test]
    fn mobile_sidechain_clone_preserves_elements() {
        let mut ms = mobile_sidechain();
        ms.elements.push(Element::N);
        ms.elements.push(Element::O);
        let cl = ms.clone();
        assert_eq!(cl.elements.as_slice(), &[Element::N, Element::O]);
    }

    #[test]
    fn system_metadata_box_vectors_none_by_default() {
        assert!(empty_metadata().box_vectors.is_none());
    }

    #[test]
    fn system_metadata_box_vectors_roundtrip() {
        let bv = [[10.0f64, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let m = SystemMetadata {
            box_vectors: Some(bv),
            ..empty_metadata()
        };
        assert_eq!(m.box_vectors, Some(bv));
    }

    #[test]
    fn system_metadata_empty_collections() {
        let m = empty_metadata();
        assert!(m.bonds.is_empty());
        assert!(m.fixed_atoms.is_empty());
        assert!(m.mobile_residues.is_empty());
    }

    #[test]
    fn system_metadata_collections_store_given_entries() {
        let m = SystemMetadata {
            bonds: vec![
                Bond {
                    a: AtomRef::Fixed(0),
                    b: AtomRef::Fixed(1),
                    order: BondOrder::Single,
                },
                Bond {
                    a: AtomRef::Fixed(1),
                    b: AtomRef::Fixed(2),
                    order: BondOrder::Aromatic,
                },
            ],
            fixed_atoms: vec![fixed_atom(), fixed_atom(), fixed_atom()],
            mobile_residues: vec![mobile_sidechain()],
            ..empty_metadata()
        };
        assert_eq!(m.bonds.len(), 2);
        assert_eq!(m.bonds[1].order, BondOrder::Aromatic);
        assert_eq!(m.fixed_atoms.len(), 3);
        assert_eq!(m.mobile_residues.len(), 1);
    }

    #[test]
    fn session_new_stores_system_and_metadata() {
        let bv = [[5.0f64, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]];
        let sess = Session::new(
            empty_system(),
            SystemMetadata {
                box_vectors: Some(bv),
                ..empty_metadata()
            },
        );
        assert_eq!(sess.system.mobile.len(), 0);
        assert_eq!(sess.metadata().box_vectors, Some(bv));
    }

    #[test]
    fn session_metadata_returns_correct_reference() {
        let bv = [[1.0f64, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let sess = Session::new(
            empty_system(),
            SystemMetadata {
                box_vectors: Some(bv),
                ..empty_metadata()
            },
        );
        assert_eq!(sess.metadata().box_vectors, Some(bv));
        assert!(sess.metadata().bonds.is_empty());
    }

    #[test]
    fn session_new_with_matching_fixed_atoms() {
        let n = 4;
        let sess = Session::new(
            system_with_n_fixed(n),
            SystemMetadata {
                fixed_atoms: vec![fixed_atom(); n],
                ..empty_metadata()
            },
        );
        assert_eq!(sess.system.fixed.positions.len(), n);
        assert_eq!(sess.metadata().fixed_atoms.len(), n);
    }

    #[test]
    fn session_clone_preserves_all_data() {
        let bv = [[7.0f64, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 7.0]];
        let sess = Session::new(
            empty_system(),
            SystemMetadata {
                box_vectors: Some(bv),
                ..empty_metadata()
            },
        );
        let cl = sess.clone();
        assert_eq!(cl.metadata().box_vectors, sess.metadata().box_vectors);
        assert_eq!(cl.system.mobile.len(), sess.system.mobile.len());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn session_new_panics_on_mobile_count_mismatch() {
        Session::new(
            empty_system(),
            SystemMetadata {
                mobile_residues: vec![mobile_sidechain()],
                ..empty_metadata()
            },
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn session_new_panics_on_fixed_count_mismatch() {
        Session::new(system_with_n_fixed(1), empty_metadata());
    }
}
