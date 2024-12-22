use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::Result;

// Import these from the top-level rerun crate, not rerun::datatypes
use rerun::{
    archetypes::Mesh3D,
    Position3D,
    RecordingStream,
    TextDocument,
    TriangleIndices,
};
// use stl_io::{self, IndexedMesh};
use urdf_rs::{self, Geometry};

/// Top-level function that:
/// 1) Reads/parses the URDF with urdf_rs
/// 2) Builds a link adjacency
/// 3) Recursively logs the hierarchy (links, visuals) to Rerun
pub fn parse_and_log_urdf_hierarchy(urdf_path: &str, rec: &RecordingStream) -> Result<()> {
    // Show the path
    dbg!(urdf_path);

    // Parse the URDF
    let robot_model = urdf_rs::read_file(urdf_path)
        .map_err(|e| anyhow::anyhow!("Failed to parse URDF at {urdf_path}: {e}"))?;

    dbg!(robot_model.links.len());
    dbg!(robot_model.joints.len());

    // The directory containing the URDF file, so we can resolve "meshes/foo.stl" etc.
    let urdf_dir = Path::new(urdf_path)
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    dbg!(&urdf_dir);

    // link_name -> Link
    let mut link_map: HashMap<String, &urdf_rs::Link> = HashMap::new();
    for link in &robot_model.links {
        link_map.insert(link.name.clone(), link);
    }

    // Build adjacency: parent -> [children]
    let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
    let mut child_links = HashSet::new();
    for joint in &robot_model.joints {
        let parent = joint.parent.link.clone();
        let child = joint.child.link.clone();
        child_links.insert(child.clone());
        adjacency.entry(parent).or_default().push(child);
    }
    dbg!(adjacency.keys().len());

    // Root links are those not used as a child anywhere
    let all_link_names: HashSet<String> = link_map.keys().cloned().collect();
    let root_links: Vec<String> = all_link_names
        .difference(&child_links)
        .cloned()
        .collect();
    dbg!(&root_links);

    // Recurse from each root
    for root in &root_links {
        log_link_recursive(
            root,
            "/robot", // or any top-level prefix
            &adjacency,
            &link_map,
            &urdf_dir,
            rec,
        )?;
    }

    Ok(())
}

/// Recursively logs each link’s data, plus each visual mesh (if STL), then recurses on children.
fn log_link_recursive(
    link_name: &str,
    parent_path: &str,
    adjacency: &HashMap<String, Vec<String>>,
    link_map: &HashMap<String, &urdf_rs::Link>,
    urdf_dir: &PathBuf,
    rec: &RecordingStream,
) -> Result<()> {

    let this_path = format!("{}/{}", parent_path, link_name);
    dbg!(&this_path);

    let link = match link_map.get(link_name) {
        Some(l) => l,
        None => {
            eprintln!("Warning: link {link_name} not found in link_map");
            return Ok(());
        }
    };

    // Build a text doc summarizing inertial & geometry
    let mut doc_text = format!("Hierarchical URDF Link: {}\n", link.name);

    // Inertial
    {
        let inertial = &link.inertial;
        doc_text.push_str(&format!("  Inertial mass: {}\n", inertial.mass.value));
        doc_text.push_str(&format!(
            "  Inertia ixx={} iyy={} izz={} ixy={} ixz={} iyz={}\n",
            inertial.inertia.ixx,
            inertial.inertia.iyy,
            inertial.inertia.izz,
            inertial.inertia.ixy,
            inertial.inertia.ixz,
            inertial.inertia.iyz
        ));
        doc_text.push_str(&format!(
            "  inertial origin xyz={:?}, rpy={:?}\n",
            inertial.origin.xyz, inertial.origin.rpy
        ));
    }

    // Visual
    if link.visual.is_empty() {
        doc_text.push_str("  (No visual geometry)\n");
    } else {
        doc_text.push_str("  Visual geometry:\n");
    }

    for (i, vis) in link.visual.iter().enumerate() {
        doc_text.push_str(&format!(
            "    #{} origin xyz={:?}, rpy={:?}\n",
            i, vis.origin.xyz, vis.origin.rpy
        ));

        match &vis.geometry {
            Geometry::Mesh { filename, scale } => {
                let rel_path = filename;
                let abs_path = urdf_dir.join(filename);
                doc_text.push_str(&format!(
                    "      Mesh Relative Path={} scale={:?}\n",
                    rel_path, scale
                ));
                doc_text.push_str(&format!(
                    "      Mesh Absolute Path={}\n",
                    abs_path.display()
                ));

                // We only handle .stl in this example
                if abs_path.extension().and_then(|e| e.to_str()) == Some("stl") {
                    match load_stl_as_mesh3d(&abs_path) {
                        Ok(mesh3d) => {
                            let mesh_entity_path = format!("{}/visual_{}/mesh", this_path, i);
                            dbg!(&mesh_entity_path);

                            // Log the mesh:
                            // *IMPORTANT* we must pass a &str to rec.log
                            // so do: mesh_entity_path.as_str()
                            rec.log(mesh_entity_path.as_str(), &mesh3d)?;
                        }
                        Err(e) => {
                            doc_text.push_str(&format!(
                                "      (Error loading STL: {e})\n"
                            ));
                        }
                    }
                } else {
                    doc_text.push_str("      (Currently only STL is handled)\n");
                }
            }
            Geometry::Box { size } => {
                doc_text.push_str(&format!("      Box size={:?}\n", size));
            }
            Geometry::Cylinder { radius, length } => {
                doc_text.push_str(&format!("      Cylinder r={} l={}\n", radius, length));
            }
            Geometry::Sphere { radius } => {
                doc_text.push_str(&format!("      Sphere r={}\n", radius));
            }
            _ => {
                doc_text.push_str("      (Other/unknown geometry)\n");
            }
        }
    }

    // Finally, log the text doc for this link
    rec.log(this_path.as_str(), &TextDocument::new(doc_text))?;

    // Recurse to children
    if let Some(children) = adjacency.get(link_name) {
        for child_link in children {
            log_link_recursive(child_link, &this_path, adjacency, link_map, urdf_dir, rec)?;
        }
    }

    Ok(())
}

/// Opens an STL file, parses it via `stl_io`, and converts to a `Mesh3D`.
fn load_stl_as_mesh3d(abs_path: &Path) -> Result<Mesh3D> {
    let file = OpenOptions::new()
        .read(true)
        .open(abs_path)
        .map_err(|e| anyhow::anyhow!("Failed to open {:?}: {e}", abs_path))?;

    let mut buf = BufReader::new(file);
    let stl = stl_io::read_stl(&mut buf)
        .map_err(|e| anyhow::anyhow!("Failed to read_stl() for {:?}: {e}", abs_path))?;

    // stl.vertices -> each is stl_io::Vertex, effectively [f32; 3].
    let positions: Vec<Position3D> = stl
        .vertices
        .iter()
        .map(|v| [v[0], v[1], v[2]].into()) // float array -> Position3D
        .collect();

    // stl.faces -> each face has .vertices: [usize; 3].
    // We must cast each usize to u32 before building TriangleIndices.
    let indices: Vec<TriangleIndices> = stl
        .faces
        .iter()
        .map(|face| {
            [
                face.vertices[0] as u32,
                face.vertices[1] as u32,
                face.vertices[2] as u32,
            ]
            .into()
        })
        .collect();

    // Construct the Rerun 3D mesh
    let mesh3d = Mesh3D::new(positions).with_triangle_indices(indices);
    mesh3d
        .sanity_check()
        .map_err(|e| anyhow::anyhow!("Mesh error: {e}"))?;

    Ok(mesh3d)
}
