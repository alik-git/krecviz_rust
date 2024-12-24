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
    TextLog,
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
        0.0, cx, -sx,
        0.0, sx,  cx,
    ];
    let r_y = [
        cy, 0.0, sy,
        0.0, 1.0, 0.0,
       -sy, 0.0, cy,
    ];
    let r_z = [
        cz, -sz, 0.0,
        sz,  cz, 0.0,
        0.0, 0.0, 1.0,
    ];

    println!("   rotation_from_euler_xyz() debug:");
    println!("     => rx={}, ry={}, rz={}", rx, ry, rz);
    println!("     => R_x (row-major) = {:?}", r_x);
    println!("     => R_y (row-major) = {:?}", r_y);
    println!("     => R_z (row-major) = {:?}", r_z);

    // Final = Rz * Ry * Rx
    let ryx = mat3x3_mul(r_y, r_x);
    let final_mat = mat3x3_mul(r_z, ryx);

    println!("     => final Rz @ Ry @ Rx (row-major) = {:?}", final_mat);
    println!("       => as a 3x3 matrix:");
    for row_i in 0..3 {
        let start = row_i * 3;
        let row_slice = &final_mat[start..(start + 3)];
        println!("         row {}: {:?}", row_i, row_slice);
    }

    final_mat
}

/// Row-major 3x3 multiply a*b
fn mat3x3_mul(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    let mut out = [0.0_f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            out[row * 3 + col] =
                a[row * 3 + 0] * b[0 * 3 + col]
                + a[row * 3 + 1] * b[1 * 3 + col]
                + a[row * 3 + 2] * b[2 * 3 + col];
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
    all_link_names.difference(&child_names).next().cloned()
}

/// Build adjacency: parent_link -> (joint_name, child_link).
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
fn get_chain(
    adjacency: &HashMap<String, Vec<(String, String)>>,
    root_link_name: &str,
    target_link_name: &str,
) -> Option<Vec<String>> {
    let mut stack = vec![(root_link_name.to_owned(), vec![root_link_name.to_owned()])];

    while let Some((cur_link, path_so_far)) = stack.pop() {
        if cur_link == target_link_name {
            return Some(path_so_far);
        }
        if let Some(children) = adjacency.get(&cur_link) {
            for (joint_name, child_link) in children {
                let mut new_path = path_so_far.clone();
                new_path.push(joint_name.clone()); // joint
                new_path.push(child_link.clone()); // link
                stack.push((child_link.clone(), new_path));
            }
        }
    }
    None
}

/// Construct the entity path for a given link, skipping the joint names in the chain [0::2].
fn link_entity_path(
    adjacency: &HashMap<String, Vec<(String, String)>>,
    root_link: &str,
    link_name: &str,
) -> Option<String> {
    if let Some(chain) = get_chain(adjacency, root_link, link_name) {
        // skip every other => link-only
        let link_names: Vec<_> = chain.iter().step_by(2).cloned().collect();
        Some(link_names.join("/"))
    } else {
        None
    }
}

/// Construct the entity path for a joint, skipping every-other item in the chain from root->child.
fn joint_entity_path(
    adjacency: &HashMap<String, Vec<(String, String)>>,
    root_link: &str,
    joint: &Joint,
) -> Option<String> {
    let child_link = &joint.child.link;
    if let Some(chain) = get_chain(adjacency, root_link, child_link) {
        let link_names: Vec<_> = chain.iter().step_by(2).cloned().collect();
        Some(link_names.join("/"))
    } else {
        None
    }
}

/// Load an STL into a Mesh3D archetype.
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
    mesh3d.sanity_check()?;

    Ok(mesh3d)
}

/// Recursively logs each link’s data and any mesh visuals, then recurses on children.
/// We still use BFS adjacency to build the link entity path, but the actual order
/// we call this function is now strictly based on `robot_model.links` iteration order.
fn log_link_meshes_in_rusts_recursive_style(
    link_name: &str,
    adjacency: &HashMap<String, Vec<(String, String)>>,
    link_map: &HashMap<String, &Link>,
    urdf_dir: &PathBuf,
    rec: &RecordingStream,
    root_link: &str,
) -> Result<()> {
    // Build the entity path (same naming approach as Python).
    let entity_path = link_entity_path(adjacency, root_link, link_name)
        .unwrap_or_else(|| link_name.to_owned());

    let this_path = &entity_path;

    // Retrieve the link from link_map
    let link = match link_map.get(link_name) {
        Some(l) => l,
        None => {
            eprintln!("Warning: link {link_name} not found in link_map");
            return Ok(());
        }
    };

    // Summarize link data in a TextDocument
    let mut doc_text = format!("Hierarchical URDF Link: {}\n", link.name);
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

    if link.visual.is_empty() {
        doc_text.push_str("  (No visual geometry)\n");
    } else {
        doc_text.push_str("  Visual geometry:\n");
    }

    // For each <visual>, we log a mesh
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

                            println!("======================");
                            println!("rerun_log");
                            println!(
                                "entity_path = entity_path with value '{}'",
                                mesh_entity_path
                            );
                            println!(
                                "entity = rerun::archetypes::Mesh3D(...) with these numeric values:"
                            );

                            let first_3_positions: Vec<[f32; 3]> = mesh3d
                                .vertex_positions
                                .iter()
                                .take(3)
                                .map(|pos| [pos.x(), pos.y(), pos.z()])
                                .collect();
                            println!("  => vertex_positions (first 3) = {:?}", first_3_positions);
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

    // Now log the link summary as a TextDocument:
    println!("======================");
    println!("rerun_log");
    println!("entity_path = this_path with value '{this_path}'");
    println!("entity = rerun::TextDocument(...)");
    println!("timeless = false");
    rec.log(this_path.as_str(), &TextDocument::new(doc_text))?;

    // We do *NOT* recursively log children here, because we want to keep the order
    // that the Python script uses: it straightforwardly iterates over links. 
    // (The BFS adjacency is only for naming the path.)

    Ok(())
}

/// Log the URDF in the same order as the Python version:
///  1) "root" entity
///  2) Joints in URDF order
///  3) Links in URDF order
pub fn parse_and_log_urdf_hierarchy(urdf_path: &str, rec: &RecordingStream) -> Result<()> {
    println!("[parse_and_log_urdf_hierarchy]  Loading URDF from: {urdf_path:?}");
    let robot_model = urdf_rs::read_file(urdf_path)
        .map_err(|e| anyhow::anyhow!("Failed to parse URDF at {urdf_path}: {e}"))?;

    println!("[parse_and_log_urdf_hierarchy]   => links.len() = {}", robot_model.links.len());
    println!("[parse_and_log_urdf_hierarchy]   => joints.len() = {}", robot_model.joints.len());

    let urdf_dir = Path::new(urdf_path)
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    let mut link_map: HashMap<String, &Link> = HashMap::new();
    for link in &robot_model.links {
        link_map.insert(link.name.clone(), link);
    }

    let adjacency = build_adjacency(&robot_model.joints);

    let root_link_name = find_root_link_name(&robot_model.links, &robot_model.joints)
        .unwrap_or_else(|| {
            eprintln!("No unique root link found! Using 'base' as fallback.");
            "base".to_owned()
        });

    // ------------------------------------------------------------
    // (A) Log the "root" coordinates at path "", same as the Python does:
    //     rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    // Since there's no direct "ViewCoordinates" in Rust, let's do a simple TextLog 
    // or a zeroed-out Transform3D. 
    // We'll just use a custom text to indicate the root is "RIGHT_HAND_Z_UP".
    // If you have a specialized component in Rust, you could log that instead.
    // ------------------------------------------------------------
    {
        println!("======================");
        println!("rerun_log");
        println!("entity_path = '' (the root path)");
        println!("entity = (Pretend) rr.ViewCoordinates.RIGHT_HAND_Z_UP");
        println!("timeless = true");

        // // We'll do a text log to mark the coordinate system:
        // rec.log("", &TextLog::info("Root coordinates => RIGHT_HAND_Z_UP"))?;
    }

    // ------------------------------------------------------------
    // (B) Now log each joint transform IN THE ORDER they appear in the URDF
    //     i.e. the direct iteration of robot_model.joints
    // (like Python does: for joint in self.urdf.joints: ...)
    // ------------------------------------------------------------
    for joint in &robot_model.joints {
        // Build entity path for the joint, using the BFS approach just for naming:
        if let Some(joint_path) = joint_entity_path(&adjacency, &root_link_name, joint) {
            // We get the joint's origin
            let origin: &Pose = &joint.origin;
            let xyz = &origin.xyz;
            let rpy = &origin.rpy;
            let (rx, ry, rz) = (rpy[0], rpy[1], rpy[2]);

            let rotation_matrix = rotation_from_euler_xyz(rx, ry, rz);

            // Flatten for debug
            let mut flattened = vec![];
            for row_i in 0..3 {
                let start = row_i * 3;
                flattened.push(rotation_matrix[start]);
                flattened.push(rotation_matrix[start + 1]);
                flattened.push(rotation_matrix[start + 2]);
            }

            println!("======================");
            println!("rerun_log");
            println!("entity_path = entity_path with value '{joint_path}'");
            println!("  => translation = {:?}", xyz);
            println!("  => rotation (full 2D) = [");
            println!("       [{:.9}, {:.9}, {:.9}],", rotation_matrix[0], rotation_matrix[1], rotation_matrix[2]);
            println!("       [{:.9}, {:.9}, {:.9}],", rotation_matrix[3], rotation_matrix[4], rotation_matrix[5]);
            println!("       [{:.9}, {:.9}, {:.9}]", rotation_matrix[6], rotation_matrix[7], rotation_matrix[8]);
            println!("     ]");
            println!("  => rotation (row-major flatten) = {:?}", flattened);

            // In Python we do:
            //   rr.Transform3D(translation=xyz, mat3x3=rotation)
            // but you can store it in .with_mat3x3(...) if you want it to propagate. 
            // We'll replicate Python exactly:
            let mut transform = Transform3D::from_translation([
                xyz[0] as f32,
                xyz[1] as f32,
                xyz[2] as f32,
            ]);
            // If you want the rotation to truly propagate in Rerun, you can do:
            transform = transform.with_mat3x3(rotation_matrix);

            println!("entity = rerun::archetypes::Transform3D(...) with value {transform:?}");
            rec.log(joint_path.as_str(), &transform)?;
        }
    }

    // ------------------------------------------------------------
    // (C) Finally, log each link’s visuals, in the order that .links appear
    //     i.e. for link in robot_model.links
    //     This matches Python’s approach: 
    //        for link in self.urdf.links:
    //            entity_path = link_entity_path(...)
    //            self.log_link(entity_path, link)
    // ------------------------------------------------------------
    for link in &robot_model.links {
        let link_name = &link.name;
        log_link_meshes_in_rusts_recursive_style(
            link_name,
            &adjacency,
            &link_map,
            &urdf_dir,
            rec,
            &root_link_name,
        )?;
    }

    Ok(())
}
