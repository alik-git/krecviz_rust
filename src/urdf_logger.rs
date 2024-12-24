use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::Result;
use rerun::{
    archetypes::{Mesh3D, Transform3D},
    components::{Position3D, TriangleIndices},
    RecordingStream,
    TextDocument,
};
use urdf_rs::{self, Geometry, Joint, Link, Pose};

/// Convert roll-pitch-yaw (XYZ Euler angles) into a row-major 3x3 rotation matrix.
fn rotation_from_euler_xyz(rx: f64, ry: f64, rz: f64) -> [f32; 9] {
    let (cx, sx) = (rx.cos() as f32, rx.sin() as f32);
    let (cy, sy) = (ry.cos() as f32, ry.sin() as f32);
    let (cz, sz) = (rz.cos() as f32, rz.sin() as f32);

    // R_x, R_y, R_z in row-major
    let r_x = [
        1.0, 0.0, 0.0,
        0.0, cx,  -sx,
        0.0, sx,   cx,
    ];
    let r_y = [
        cy,  0.0, sy,
        0.0, 1.0, 0.0,
       -sy,  0.0, cy,
    ];
    let r_z = [
        cz, -sz, 0.0,
        sz,  cz, 0.0,
        0.0, 0.0, 1.0,
    ];

    // Multiply them: Rz * Ry * Rx
    let ryx = mat3x3_mul(r_y, r_x);
    mat3x3_mul(r_z, ryx)
}

/// Row-major 3x3 multiply a*b
fn mat3x3_mul(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    let mut out = [0.0_f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            out[row * 3 + col] =
                a[row * 3 + 0] * b[0 * 3 + col] +
                a[row * 3 + 1] * b[1 * 3 + col] +
                a[row * 3 + 2] * b[2 * 3 + col];
        }
    }
    out
}

/// Return the name of the root link (the one that doesn't appear as a child in any joint).
fn find_root_link_name(links: &[Link], joints: &[Joint]) -> Option<String> {
    let mut all_link_names = HashSet::new();
    let mut child_names = HashSet::new();
    for link in links {
        all_link_names.insert(link.name.clone());
    }
    for joint in joints {
        child_names.insert(joint.child.link.clone());
    }
    // The root is a link name that is never a child
    all_link_names
        .difference(&child_names)
        .next()
        .cloned()
}

/// Build adjacency: parent_link -> (joint_name, child_link).
///
/// We'll store enough info so we can do a BFS or DFS to reconstruct the chain from root->child.
fn build_adjacency(joints: &[Joint]) -> HashMap<String, Vec<(String, String)>> {
    let mut adj = HashMap::new();
    for joint in joints {
        let parent_link_name = &joint.parent.link;
        let joint_name = &joint.name;
        let child_link_name = &joint.child.link;

        adj.entry(parent_link_name.clone())
            .or_insert_with(Vec::new)
            .push((joint_name.clone(), child_link_name.clone()));
    }
    adj
}

/// Return a chain of names [link0, joint0, link1, joint1, link2, …]
/// from the root_link_name to `target_link_name`.
/// E.g. the Python `.get_chain("base", "arm2_shell_2")` might return
/// ['base','floating_base','body1-part','Revolute_1','shoulder_2','Revolute_3','arm1_top_2','Revolute_6','arm2_shell_2'].
fn get_chain(
    adjacency: &HashMap<String, Vec<(String, String)>>,
    root_link_name: &str,
    target_link_name: &str,
) -> Option<Vec<String>> {
    // We'll do a DFS storing the path as we go:
    // Each node in the stack: (current_link, path_so_far)
    let mut stack = vec![(root_link_name.to_owned(), vec![root_link_name.to_owned()])];

    while let Some((cur_link, path_so_far)) = stack.pop() {
        if cur_link == target_link_name {
            return Some(path_so_far);
        }
        // Explore children
        if let Some(children) = adjacency.get(&cur_link) {
            for (joint_name, child_link) in children {
                let mut new_path = path_so_far.clone();
                new_path.push(joint_name.clone());    // joint
                new_path.push(child_link.clone());    // link
                stack.push((child_link.clone(), new_path));
            }
        }
    }
    None
}

/// Construct the entity path for a given link, by skipping the joint names in the chain: [0::2].
fn link_entity_path(
    adjacency: &HashMap<String, Vec<(String, String)>>,
    root_link: &str,
    link_name: &str,
) -> Option<String> {
    if let Some(chain) = get_chain(adjacency, root_link, link_name) {
        // Skip every other item => link-only
        let link_names: Vec<_> = chain.iter().step_by(2).cloned().collect();
        Some(link_names.join("/"))
    } else {
        None
    }
}

/// Construct the entity path for a given joint: we find the chain from root->child link,
/// then skip every other item. (Because the child link name is how Python does it.)
fn joint_entity_path(
    adjacency: &HashMap<String, Vec<(String, String)>>,
    root_link: &str,
    joint: &Joint,
) -> Option<String> {
    let child_link = &joint.child.link;
    if let Some(chain) = get_chain(adjacency, root_link, child_link) {
        // skip every-other item => link-only
        let link_names: Vec<_> = chain.iter().step_by(2).cloned().collect();
        Some(link_names.join("/"))
    } else {
        None
    }
}

/// Load an STL into a Mesh3D archetype
fn load_stl_as_mesh3d(abs_path: &Path) -> Result<Mesh3D> {
    let file = OpenOptions::new()
        .read(true)
        .open(abs_path)
        .map_err(|e| anyhow::anyhow!("Failed to open {:?}: {e}", abs_path))?;

    let mut buf = BufReader::new(file);
    let stl = stl_io::read_stl(&mut buf)
        .map_err(|e| anyhow::anyhow!("Failed to read_stl() for {:?}: {e}", abs_path))?;

    let positions: Vec<Position3D> = stl
        .vertices
        .iter()
        .map(|v| Position3D::from([v[0], v[1], v[2]]))
        .collect();

    // Must specify Vec<TriangleIndices> or it won't compile
    let indices: Vec<TriangleIndices> = stl
        .faces
        .iter()
        .map(|face| {
            TriangleIndices::from([
                face.vertices[0] as u32,
                face.vertices[1] as u32,
                face.vertices[2] as u32,
            ])
        })
        .collect();

    let mesh3d = Mesh3D::new(positions).with_triangle_indices(indices);
    mesh3d
        .sanity_check()
        .map_err(|e| anyhow::anyhow!("Mesh error: {e}"))?;

    Ok(mesh3d)
}

/// Recursively logs each link’s data (TextDocument) + any mesh visuals,
/// then recurses on children.  We replicate the "link_entity_path" usage from Python.
fn log_link_recursive(
    link_name: &str,
    adjacency: &HashMap<String, Vec<(String, String)>>,
    link_map: &HashMap<String, &Link>,
    urdf_dir: &PathBuf,
    rec: &RecordingStream,
    root_link: &str,
) -> Result<()> {
    // Build the entity path
    let entity_path = link_entity_path(adjacency, root_link, link_name)
        .unwrap_or_else(|| link_name.to_owned());

    let this_path = entity_path; // we'll reuse the name 'this_path'

    // Debug
    dbg!(&this_path);

    // Grab the link
    let link = match link_map.get(link_name) {
        Some(l) => l,
        None => {
            eprintln!("Warning: link {link_name} not found in link_map");
            return Ok(());
        }
    };

    // Summarize
    let mut doc_text = format!("Hierarchical URDF Link: {}\n", link.name);

    // Inertial
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

    // Visual
    if link.visual.is_empty() {
        doc_text.push_str("  (No visual geometry)\n");
    } else {
        doc_text.push_str("  Visual geometry:\n");
    }

    // For each visual, build the mesh entity path
    for (i, vis) in link.visual.iter().enumerate() {
        doc_text.push_str(&format!(
            "    #{} origin xyz={:?}, rpy={:?}\n",
            i, vis.origin.xyz, vis.origin.rpy
        ));

        match &vis.geometry {
            Geometry::Mesh { filename, scale } => {
                let abs_path = urdf_dir.join(filename);
                doc_text.push_str(&format!(
                    "      Mesh Relative Path={} scale={:?}\n",
                    filename, scale
                ));
                doc_text.push_str(&format!("      Mesh Absolute Path={}\n", abs_path.display()));

                if abs_path.extension().and_then(|e| e.to_str()) == Some("stl") {
                    match load_stl_as_mesh3d(&abs_path) {
                        Ok(mesh3d) => {
                            let mesh_entity_path = format!("{}/visual_{}", this_path, i);

                            // Print debug info (like Python)
                            println!("======================");
                            println!("rerun_log");
                            println!(
                                "entity_path = entity_path with value '{}'",
                                mesh_entity_path
                            );
                            println!("entity = rerun::archetypes::Mesh3D(...) with these numeric values:");

                            let first_3_positions: Vec<[f32; 3]> = mesh3d
                                .vertex_positions
                                .iter()
                                .take(3)
                                .map(|pos| [pos.x(), pos.y(), pos.z()])
                                .collect();
                            println!(
                                "  => vertex_positions (first 3) = {:?}",
                                first_3_positions
                            );
                            println!("timeless = true");

                            rec.log(mesh_entity_path.as_str(), &mesh3d)?;
                        }
                        Err(e) => {
                            doc_text.push_str(&format!("      (Error loading STL: {e})\n"));
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

    // Debug prints for the text doc
    println!("======================");
    println!("rerun_log");
    println!("entity_path = this_path with value '{this_path}'");
    println!("entity = rerun::TextDocument(...)");
    println!("timeless = false");

    rec.log(this_path.as_str(), &TextDocument::new(doc_text))?;

    // Recurse on children
    if let Some(children) = adjacency.get(link_name) {
        for (joint_name, child_link_name) in children {
            // We only care about the child link
            log_link_recursive(child_link_name, adjacency, link_map, urdf_dir, rec, root_link)?;
        }
    }

    Ok(())
}

/// Log each joint transform at the same path as the final link path in Python.
/// Then recursively log each link’s mesh at the same path as well.
pub fn parse_and_log_urdf_hierarchy(urdf_path: &str, rec: &RecordingStream) -> Result<()> {
    dbg!(urdf_path);

    let robot_model = urdf_rs::read_file(urdf_path)
        .map_err(|e| anyhow::anyhow!("Failed to parse URDF at {urdf_path}: {e}"))?;
    dbg!(robot_model.links.len());
    dbg!(robot_model.joints.len());

    let urdf_dir = Path::new(urdf_path)
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    dbg!(&urdf_dir);

    // Build link map
    let mut link_map: HashMap<String, &Link> = HashMap::new();
    for link in &robot_model.links {
        link_map.insert(link.name.clone(), link);
    }

    // Build adjacency
    let adjacency = build_adjacency(&robot_model.joints);

    // Find root
    let root_link_name =
        find_root_link_name(&robot_model.links, &robot_model.joints).unwrap_or_else(|| {
            eprintln!("No unique root link found!");
            "base".to_owned()
        });
    dbg!(&root_link_name);

    // ----
    // (A) Log each joint's transform at the final link path
    // (like python: the path is the chain from root->child [0::2]).
    // ----
    for joint in &robot_model.joints {
        // We'll do the same step: chain from root->joint.child
        if let Some(chain) = get_chain(&adjacency, &root_link_name, &joint.child.link) {
            // skip every-other => link-only
            let link_names: Vec<_> = chain.iter().step_by(2).cloned().collect();
            let entity_path_w_prefix = link_names.join("/");

            let origin: &Pose = &joint.origin;
            let xyz = &origin.xyz;
            let rpy = &origin.rpy;
            let (rx, ry, rz) = (rpy[0], rpy[1], rpy[2]);
            let rotation_matrix = rotation_from_euler_xyz(rx, ry, rz);

            // Debug prints
            println!("======================");
            println!("rerun_log");
            println!("entity_path = entity_path_w_prefix with value '{entity_path_w_prefix}'");
            println!("translation = {:?}", xyz);
            println!("rotation = {:?}", rotation_matrix);

            let transform = Transform3D::from_translation([
                xyz[0] as f32,
                xyz[1] as f32,
                xyz[2] as f32,
            ])
            .with_mat3x3(rotation_matrix);

            println!("entity = rerun::archetypes::Transform3D(...) with value {transform:?}");

            rec.log(entity_path_w_prefix.as_str(), &transform)?;
        }
    }

    // ----
    // (B) Recursively log each link’s mesh
    // ----
    // Just find the top-level root(s) and traverse
    log_link_recursive(
        &root_link_name,
        &adjacency,
        &link_map,
        &urdf_dir,
        rec,
        &root_link_name,
    )?;

    Ok(())
}