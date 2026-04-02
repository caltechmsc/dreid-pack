use bio_forge::{Element, Structure};

pub fn sidechain(
    crystal: &Structure,
    packed: &Structure,
    chain_id: &str,
    res_id: i32,
    ins_code: Option<char>,
) -> Option<f64> {
    let res_c = crystal.find_residue(chain_id, res_id, ins_code)?;
    let res_p = packed.find_residue(chain_id, res_id, ins_code)?;

    let mut sum_sq = 0.0;
    let mut count = 0u32;

    for atom_c in res_c.atoms() {
        if is_backbone(&atom_c.name) || atom_c.element == Element::H {
            continue;
        }
        if let Some(atom_p) = res_p.atom(&atom_c.name) {
            let dx = atom_c.pos.x - atom_p.pos.x;
            let dy = atom_c.pos.y - atom_p.pos.y;
            let dz = atom_c.pos.z - atom_p.pos.z;
            sum_sq += dx * dx + dy * dy + dz * dz;
            count += 1;
        }
    }

    if count == 0 {
        return None;
    }

    Some((sum_sq / count as f64).sqrt())
}

fn is_backbone(name: &str) -> bool {
    matches!(
        name,
        "N" | "CA" | "C" | "O" | "OXT" | "H" | "HA" | "HA2" | "HA3"
    )
}
