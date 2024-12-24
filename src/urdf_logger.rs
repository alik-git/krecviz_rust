use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::Result;
use rerun::{
    archetypes::{Mesh3D, Transform3D},
    components::{Blob as _, Blob, ImageBuffer, ImageFormat, Position3D, TriangleIndices},
    RecordingStream,
    TextDocument,
    TextLog,
};
use urdf_rs::{self, Color, Geometry, Joint, Link, Material, Pose};

use nalgebra as na;
use parry3d::shape::{Ball as ParrySphere, Cuboid, Cylinder as ParryCylinder};
use image; // for loading actual images

/// Minimal info (color & texture path) from a URDF Material.
#[derive(Default, Debug)]
struct RrMaterialInfo {
    /// RGBA in [0..1].
    color_rgba: Option<[f32; 4]>,
    /// Absolute path to a texture file, if any.
    texture_path: Option<PathBuf>,
}

/// Convert Euler angles to row-major 3x3 rotation matrix.
/// Now with an EXTRA manual 90° rotation about X appended at the end.
fn rotation_from_euler_xyz(rx: f64, ry: f64, rz: f64) -> [f32; 9] {
    let (cx, sx) = (rx.cos() as f32, rx.sin() as f32);
    let (cy, sy) = (ry.cos() as f32, ry.sin() as f32);
    let (cz, sz) = (rz.cos() as f32, rz.sin() as f32);

    let r_x = [
        1.0, 0.0,  0.0,
        0.0, cx,  -sx,
        0.0, sx,   cx,
    ];

    let r_y = [
        cy,  0.0,  sy,
        0.0, 1.0,  0.0,
       -sy,  0.0,  cy,
    ];

    let r_z = [
        cz, -sz,  0.0,
        sz,  cz,  0.0,
        0.0, 0.0, 1.0,
    ];

    // For illustration, pick some multiply order:
    // final_mat = Rz * Ry * Rx
    let ryx = mat3x3_mul(r_y, r_x);
    let mut final_mat = mat3x3_mul(r_z, ryx);

    // -------------------------------------------------
    // EXTRA SHIFT: a manual +90° about X
    //   i.e. angle = +pi/2
    // In row-major, that is:
    //   [1,  0,  0,
    //    0,  0, -1,
    //    0,  1,  0]
    // Let’s call it manual_rx_90:
    let manual_rx_90 = [
         1.4,  0.3,  0.0,
         0.0,  1.6, -1.0,
         0.1,  -0.3,  -1.2,
    ];
    // Multiply it in (depending on which side you want to rotate from).
    // E.g. post-multiply:
    final_mat = mat3x3_mul(manual_rx_90, final_mat);
    // -------------------------------------------------

    // Print a debug snippet:
    println!("=== rotation_from_euler_xyz() debug ===");
    println!("rx = {}, ry = {}, rz = {}", rx, ry, rz);
    println!("r_x (row-major) = {:?}", r_x);
    println!("r_y (row-major) = {:?}", r_y);
    println!("r_z (row-major) = {:?}", r_z);
    println!("final_mat BEFORE manual +90° about X (RzRyRx) = {:?}", mat3x3_mul(r_z, ryx));
    println!("manual_rx_90 = {:?}", manual_rx_90);
    println!("final_mat AFTER manual +90° about X = {:?}", final_mat);
    println!("========================================");

    final_mat
}

/// Multiply row-major 3x3 a*b
fn mat3x3_mul(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    let mut out = [0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            out[row * 3 + col] =
                a[row * 3 + 0] * b[col + 0] +
                a[row * 3 + 1] * b[col + 3] +
                a[row * 3 + 2] * b[col + 6];
        }
    }
    out
}

/// Build adjacency: parent_link -> Vec<(joint_name, child_link)>
fn build_adjacency(joints: &[Joint]) -> HashMap<String, Vec<(String, String)>> {
    let mut adj = HashMap::new();
    for j in joints {
        let parent_link = j.parent.link.clone();
        let child_link = j.child.link.clone();
        adj.entry(parent_link)
            .or_insert_with(Vec::new)
            .push((j.name.clone(), child_link));
    }
    adj
}

/// Return name of the root link: the link that never appears as a child
fn find_root_link_name(links: &[Link], joints: &[Joint]) -> Option<String> {
    let mut all_links = HashSet::new();
    let mut child_links = HashSet::new();
    for l in links {
        all_links.insert(l.name.clone());
    }
    for j in joints {
        child_links.insert(j.child.link.clone());
    }
    all_links.difference(&child_links).next().cloned()
}

/// BFS approach: gather chain [link0, joint0, link1, joint1, link2,…]
fn get_chain(
    adjacency: &HashMap<String, Vec<(String, String)>>,
    root_link: &str,
    target_link: &str,
) -> Option<Vec<String>> {
    let mut stack = vec![(root_link.to_owned(), vec![root_link.to_owned()])];
    while let Some((cur_link, path_so_far)) = stack.pop() {
        if cur_link == target_link {
            return Some(path_so_far);
        }
        if let Some(children) = adjacency.get(&cur_link) {
            for (joint_name, child_link) in children {
                let mut new_path = path_so_far.clone();
                new_path.push(joint_name.clone());
                new_path.push(child_link.clone());
                stack.push((child_link.clone(), new_path));
            }
        }
    }
    None
}

/// Construct entity path for a link, skipping joints in BFS chain.
fn link_entity_path(
    adjacency: &HashMap<String, Vec<(String, String)>>,
    root_link: &str,
    link_name: &str,
) -> Option<String> {
    if let Some(chain) = get_chain(adjacency, root_link, link_name) {
        let link_names: Vec<_> = chain.iter().step_by(2).cloned().collect();
        Some(link_names.join("/"))
    } else {
        None
    }
}

/// Construct entity path for a joint, skipping every-other item in BFS chain.
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

/// Load .stl as Mesh3D
fn load_stl_as_mesh3d(abs_path: &Path) -> Result<Mesh3D> {
    let f = OpenOptions::new()
        .read(true)
        .open(abs_path)
        .map_err(|e| anyhow::anyhow!("Failed to open {abs_path:?}: {e}"))?;
    let mut buf = BufReader::new(f);
    let stl = stl_io::read_stl(&mut buf)
        .map_err(|e| anyhow::anyhow!("stl_io error reading {abs_path:?}: {e}"))?;

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

    let mesh = Mesh3D::new(positions).with_triangle_indices(indices);
    mesh.sanity_check()?;
    Ok(mesh)
}

/// Parse color/texture from a URDF <material>
fn parse_urdf_material(mat: &Material, urdf_dir: &Path) -> RrMaterialInfo {
    let mut info = RrMaterialInfo::default();
    // if <color> is present
    if let Some(c) = &mat.color {
        // Vec4 derefs to [f64; 4], so we can index it directly
        let rgba = &*c.rgba; // dereference to get the array
        info.color_rgba = Some([
            rgba[0] as f32,
            rgba[1] as f32,
            rgba[2] as f32,
            rgba[3] as f32,
        ]);
    }

    // if <texture> is present
    if let Some(tex) = &mat.texture {
        let abs = urdf_dir.join(&tex.filename);
        if abs.exists() {
            info.texture_path = Some(abs);
        }
    }

    info
}

/// Convert float RGBA -> u8 RGBA in [0..255]
fn float_rgba_to_u8(rgba: [f32; 4]) -> [u8; 4] {
    [
        (rgba[0] * 255.0).clamp(0.0, 255.0) as u8,
        (rgba[1] * 255.0).clamp(0.0, 255.0) as u8,
        (rgba[2] * 255.0).clamp(0.0, 255.0) as u8,
        (rgba[3] * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

/// Load an image from disk → Rerun ImageBuffer
fn load_image_as_rerun_buffer(path: &Path) -> Result<ImageBuffer> {
    let img = image::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open {path:?}: {e}"))?;
    let rgba8 = img.to_rgba8().into_raw(); // Vec<u8>

    // Rerun's data blob:
    let data_blob: rerun::datatypes::Blob = rgba8.into();
    let image_buf = ImageBuffer(data_blob);
    Ok(image_buf)
}

/// The main BFS-based link geometry logging.  
/// We replicate Python logic to handle global named materials too!
fn log_link_meshes_in_rusts_recursive_style(
    link_name: &str,
    adjacency: &HashMap<String, Vec<(String, String)>>,
    link_map: &HashMap<String, &Link>,
    urdf_dir: &PathBuf,
    rec: &RecordingStream,
    root_link: &str,
    all_materials_map: &HashMap<String, &Material>, // newly added
) -> Result<()> {
    let entity_path = link_entity_path(adjacency, root_link, link_name)
        .unwrap_or_else(|| link_name.to_owned());

    let link = match link_map.get(link_name) {
        Some(l) => l,
        None => {
            eprintln!("Warning: link {link_name} not found in link_map!");
            return Ok(());
        }
    };

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

    for (i, vis) in link.visual.iter().enumerate() {
        doc_text.push_str(&format!(
            "    #{} origin xyz={:?}, rpy={:?}\n",
            i, vis.origin.xyz, vis.origin.rpy
        ));

        // Step (1): gather material info.
        let mut mat_info = RrMaterialInfo::default();
        if let Some(vis_mat) = &vis.material {
            // If inline <material> has no <color> or <texture> => it's a reference to global <material name="..."/>
            let mat_name = &vis_mat.name; // e.g. "blue"
            if vis_mat.color.is_none() && vis_mat.texture.is_none() {
                // Look up global
                if let Some(global_mat) = all_materials_map.get(mat_name) {
                    mat_info = parse_urdf_material(global_mat, urdf_dir);
                }
            } else {
                // inline color or texture
                mat_info = parse_urdf_material(vis_mat, urdf_dir);
            }
        }

        // Step (2): build geometry
        let mesh_entity_path = format!("{}/visual_{}", entity_path, i);
        let (mesh3d, extra_txt) = match &vis.geometry {
            Geometry::Mesh { filename, scale } => {
                let abs_path = urdf_dir.join(filename);
                let mut txt = format!("      Mesh file={:?}, scale={scale:?}\n", abs_path);
                if abs_path.extension().and_then(|e| e.to_str()) == Some("stl") {
                    match load_stl_as_mesh3d(&abs_path) {
                        Ok(m) => (m, txt),
                        Err(e) => {
                            txt.push_str(&format!("(Error loading STL: {e})\n"));
                            (Mesh3D::new(Vec::<[f32; 3]>::new()), txt)
                        }
                    }
                } else {
                    txt.push_str("      (Currently only .stl is handled)\n");
                    (Mesh3D::new(Vec::<[f32; 3]>::new()), txt)
                }
            }
            Geometry::Box { size } => {
                let (sx, sy, sz) = (size[0], size[1], size[2]);
                let msg = format!("      Box size=({},{},{})\n", sx, sy, sz);
                let cuboid = Cuboid::new(na::Vector3::new(
                    (sx / 2.0) as f32,
                    (sy / 2.0) as f32,
                    (sz / 2.0) as f32,
                ));
                let (raw_v, raw_i) = cuboid.to_trimesh();
                let positions: Vec<Position3D> = raw_v
                    .iter()
                    .map(|p| Position3D::new(p.x, p.y, p.z))
                    .collect();
                let tri_idxs: Vec<TriangleIndices> = raw_i
                    .iter()
                    .map(|[a,b,c]| TriangleIndices::from([*a, *b, *c]))
                    .collect();
                let mesh = Mesh3D::new(positions).with_triangle_indices(tri_idxs);
                (mesh, msg)
            }
            Geometry::Cylinder { radius, length } => {
                let msg = format!("      Cylinder radius={}, length={}\n", radius, length);
                let half_height = (*length as f32) / 2.0;
                let cyl = ParryCylinder::new(half_height, *radius as f32);
                let (raw_v, raw_i) = cyl.to_trimesh(30);
                let positions: Vec<Position3D> = raw_v
                    .iter()
                    .map(|p| Position3D::new(p.x, p.y, p.z))
                    .collect();
                let tri_idxs: Vec<TriangleIndices> = raw_i
                    .iter()
                    .map(|[a,b,c]| TriangleIndices::from([*a, *b, *c]))
                    .collect();
                let mesh = Mesh3D::new(positions).with_triangle_indices(tri_idxs);
                (mesh, msg)
            }
            Geometry::Sphere { radius } => {
                let msg = format!("      Sphere radius={}\n", radius);
                let ball = ParrySphere::new(*radius as f32);
                let (raw_v, raw_i) = ball.to_trimesh(20, 20);
                let positions: Vec<Position3D> = raw_v
                    .iter()
                    .map(|p| Position3D::new(p.x, p.y, p.z))
                    .collect();
                let tri_idxs: Vec<TriangleIndices> = raw_i
                    .iter()
                    .map(|[a,b,c]| TriangleIndices::from([*a, *b, *c]))
                    .collect();
                let mesh = Mesh3D::new(positions).with_triangle_indices(tri_idxs);
                (mesh, msg)
            }
            _ => {
                let msg = String::from("      (Unsupported geometry)\n");
                (Mesh3D::new(Vec::<[f32; 3]>::new()), msg)
            }
        };
        doc_text.push_str(&extra_txt);

        // Step (3): apply color or texture
        let mut final_mesh = mesh3d;
        if let Some(rgba) = mat_info.color_rgba {
            let col_u8 = float_rgba_to_u8(rgba);
            let n_verts = final_mesh.vertex_positions.len();
            let mut all_colors = Vec::with_capacity(n_verts);
            for _ in 0..n_verts {
                all_colors.push(col_u8);
            }
            final_mesh = final_mesh.with_vertex_colors(all_colors);
        }
        if let Some(tex_path) = &mat_info.texture_path {
            match load_image_as_rerun_buffer(tex_path) {
                Ok(img_buf) => {
                    // We'll just do a dummy dimension:
                    // If you want correct dimension, do `let (w,h) = image::image_dimensions(tex_path)?;`
                    let (w, h) = image::image_dimensions(tex_path).unwrap_or((1,1));
                    let format = ImageFormat::rgba8([w, h]);
                    final_mesh = final_mesh.with_albedo_texture(format, img_buf);
                }
                Err(e) => eprintln!("Warning: failed to load texture {tex_path:?}: {e}"),
            }
        }

        // Step (4): log
        println!("======================");
        println!("rerun_log");
        println!("entity_path = '{}'", mesh_entity_path);
        println!(" => geometry has {} vertices", final_mesh.vertex_positions.len());
        rec.log(mesh_entity_path.as_str(), &final_mesh)?;
    }

    // Summarize link
    let link_entity_path = entity_path.as_str();
    println!("======================");
    println!("rerun_log");
    println!("entity_path = '{}'", link_entity_path);
    println!("entity = rerun::TextDocument(...)");
    rec.log(link_entity_path, &TextDocument::new(doc_text))?;

    Ok(())
}

/// The main function that replicates the Python logic:
///  1) root
///  2) each joint transform
///  3) each link’s geometry
pub fn parse_and_log_urdf_hierarchy(urdf_path: &str, rec: &RecordingStream) -> Result<()> {
    let robot_model = urdf_rs::read_file(urdf_path)
        .map_err(|e| anyhow::anyhow!("Failed to parse URDF {urdf_path:?}: {e}"))?;

    // Build adjacency
    let adjacency = build_adjacency(&robot_model.joints);

    // Find root
    let root_link_name = find_root_link_name(&robot_model.links, &robot_model.joints)
        .unwrap_or_else(|| "base".to_owned());

    // Collect global named materials:
    let mut all_materials_map = HashMap::new();
    for m in &robot_model.materials {
        all_materials_map.insert(m.name.clone(), m);
    }

    // (A) Log root
    println!("======================");
    println!("rerun_log");
    println!("entity_path = '' (root path)");
    println!("entity = (Pretend) rr.ViewCoordinates.RIGHT_HAND_Z_UP");
    println!("timeless = true");

    // (B) Log each joint transform
    for j in &robot_model.joints {
        if let Some(joint_path) = joint_entity_path(&adjacency, &root_link_name, j) {
            let xyz = j.origin.xyz;
            let rpy = j.origin.rpy;
            let mat = rotation_from_euler_xyz(rpy[0], rpy[1], rpy[2]);
            let mut tf = Transform3D::from_translation([xyz[0] as f32, xyz[1] as f32, xyz[2] as f32]);
            tf = tf.with_mat3x3(mat);

            println!("======================");
            println!("rerun_log");
            println!("entity_path = '{}'", joint_path);
            println!(" => translation={xyz:?}, rotation= {mat:?}");
            rec.log(joint_path.as_str(), &tf)?;
        }
    }

    // (C) For each link, BFS log geometry
    let mut link_map: HashMap<String, &Link> = HashMap::new();
    for l in &robot_model.links {
        link_map.insert(l.name.clone(), l);
    }

    let urdf_dir = Path::new(urdf_path).parent().map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    for link in &robot_model.links {
        log_link_meshes_in_rusts_recursive_style(
            &link.name,
            &adjacency,
            &link_map,
            &urdf_dir,
            rec,
            &root_link_name,
            &all_materials_map, // pass global named materials
        )?;
    }

    Ok(())
}
