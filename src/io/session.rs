use crate::model::system::{MAX_SIDECHAIN_ATOMS, System};
use arrayvec::ArrayVec;

pub use dreid_forge::{BondOrder, Element, ResidueCategory, ResiduePosition, StandardResidue};

/// A packing session: the molecular system together with its topology and metadata.
#[derive(Debug, Clone)]
pub struct Session {
    /// Packable system (mobile residues + fixed scaffold).
    pub system: System,
    // Topology + metadata bookkeeping for biological system.
    #[allow(dead_code)] // FIXME: Remove allow dead code once metadata is used.
    metadata: SystemMetadata,
}

impl Session {
    /// Creates a session from a `system` and its paired `metadata`.
    #[allow(dead_code)] // FIXME: Remove allow dead code once this is used.
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
    #[allow(dead_code)] // FIXME: Remove allow dead code once this is used.
    pub(super) fn metadata(&self) -> &SystemMetadata {
        &self.metadata
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
