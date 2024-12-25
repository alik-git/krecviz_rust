use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::OpenOptions;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::Result;
use rerun::{
    archetypes::{Mesh3D},
    components::{Blob as _, Blob, ImageBuffer, ImageFormat, Position3D, TriangleIndices},
    RecordingStream,
    TextDocument,
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

// ----------------------------------------------------------------------------
// Utilities for 3×3 & 4×4 transforms

/// Convert Euler angles (rx, ry, rz) to row-major 3x3 rotation matrix: final_mat = Rz * Ry * Rx.
fn rotation_from_euler_xyz(rx: f64, ry: f64, rz: f64) -> [f32; 9] {
    let (cx, sx) = (rx.cos() as f32, rx.sin() as f32);
    let (cy, sy) = (ry.cos() as f32, ry.sin() as f32);
    let (cz, sz) = (rz.cos() as f32, rz.sin() as f32);

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

    let ryx = mat3x3_mul(r_y, r_x);
    mat3x3_mul(r_z, ryx)
}

/// Multiply two row-major 3x3 matrices (a*b).
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

/// Build a row-major 4x4 transform from xyz + rpy (Euler angles).
fn build_4x4_from_xyz_rpy(xyz: [f64; 3], rpy: [f64; 3]) -> [f32; 16] {
    let rot3x3 = rotation_from_euler_xyz(rpy[0], rpy[1], rpy[2]);
    [
        rot3x3[0], rot3x3[1], rot3x3[2], xyz[0] as f32,
        rot3x3[3], rot3x3[4], rot3x3[5], xyz[1] as f32,
        rot3x3[6], rot3x3[7], rot3x3[8], xyz[2] as f32,
        0.0,       0.0,       0.0,       1.0,
    ]
}

/// Multiply two row-major 4x4 matrices (a*b).
fn mat4x4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for row in 0..4 {
        for col in 0..4 {
            let mut val = 0.0;
            for k in 0..4 {
                val += a[row * 4 + k] * b[k * 4 + col];
            }
            out[row * 4 + col] = val;
        }
    }
    out
}

// ----------------------------------------------------------------------------
// Baked transform for a Mesh3D

/// Like Python `mesh.apply_transform(transform)`, applying a 4x4 in row-major to vertex positions (and normals).
fn apply_4x4_to_mesh3d(mesh: &mut Mesh3D, transform: [f32; 16]) {
    // Positions: treat as (x,y,z,1)
    for vertex in &mut mesh.vertex_positions {
        let x = vertex[0];
        let y = vertex[1];
        let z = vertex[2];
        let w = 1.0;
        let xp = transform[0] * x + transform[1] * y + transform[2] * z + transform[3] * w;
        let yp = transform[4] * x + transform[5] * y + transform[6] * z + transform[7] * w;
        let zp = transform[8] * x + transform[9] * y + transform[10] * z + transform[11] * w;
        vertex[0] = xp;
        vertex[1] = yp;
        vertex[2] = zp;
    }
    // Normals: treat as (nx, ny, nz, 0) (no translation)
    if let Some(ref mut normals) = mesh.vertex_normals {
        for normal in normals {
            let nx = normal[0];
            let ny = normal[1];
            let nz = normal[2];
            let w = 0.0;
            let nxp = transform[0] * nx + transform[1] * ny + transform[2] * nz + transform[3] * w;
            let nyp = transform[4] * nx + transform[5] * ny + transform[6] * nz + transform[7] * w;
            let nzp = transform[8] * nx + transform[9] * ny + transform[10] * nz + transform[11] * w;
            normal[0] = nxp;
            normal[1] = nyp;
            normal[2] = nzp;
        }
    }
}

// ----------------------------------------------------------------------------
// Adjacency + BFS to compute global transforms

/// Build adjacency: parent_link -> Vec<(joint, child_link)>
fn build_adjacency(joints: &[Joint]) -> HashMap<String, Vec<(Joint, String)>> {
    let mut adj = HashMap::new();
    for j in joints {
        let parent_link_name = j.parent.link.clone();
        let child_link_name = j.child.link.clone();
        adj.entry(parent_link_name)
            .or_insert_with(Vec::new)
            .push((j.clone(), child_link_name));
    }
    adj
}

/// Find the link that never appears as a child → typically the root.
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

/// For each link, compute its global transform from the root by chaining all the joint transforms.
fn build_link_global_transforms(
    adjacency: &HashMap<String, Vec<(Joint, String)>>,
    root_link_name: &str,
    joints: &[Joint],
) -> HashMap<String, [f32; 16]> {
    let mut link_to_tf = HashMap::new();

    // The root link gets an identity transform
    link_to_tf.insert(root_link_name.to_owned(), [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]);

    // BFS queue: (current_link_name)
    let mut queue = VecDeque::new();
    queue.push_back(root_link_name.to_owned());

    while let Some(cur_link) = queue.pop_front() {
        let cur_tf = link_to_tf.get(&cur_link).cloned().unwrap();

        if let Some(child_joints) = adjacency.get(&cur_link) {
            for (j, child_link_name) in child_joints {
                // Convert Vec3 to arrays for xyz and rpy
                let xyz = [j.origin.xyz[0], j.origin.xyz[1], j.origin.xyz[2]];
                let rpy = [j.origin.rpy[0], j.origin.rpy[1], j.origin.rpy[2]];
                let local_tf = build_4x4_from_xyz_rpy(xyz, rpy);
                let child_tf = mat4x4_mul(cur_tf, local_tf);
                link_to_tf.insert(child_link_name.clone(), child_tf);

                // Enqueue child
                queue.push_back(child_link_name.clone());
            }
        }
    }

    link_to_tf
}

// ----------------------------------------------------------------------------
// For building the BFS chain from root -> link to get entity path for logging

/// BFS-based approach to gather [link0, joint0, link1, joint1, link2,…] to build a path string
fn get_chain(
    adjacency: &HashMap<String, Vec<(Joint, String)>>,
    root_link: &str,
    target_link: &str,
) -> Option<Vec<String>> {
    let mut stack = vec![(root_link.to_owned(), vec![root_link.to_owned()])];
    while let Some((cur_link, path_so_far)) = stack.pop() {
        if cur_link == target_link {
            return Some(path_so_far);
        }
        if let Some(children) = adjacency.get(&cur_link) {
            for (joint, child_link) in children {
                let mut new_path = path_so_far.clone();
                new_path.push(joint.name.clone());
                new_path.push(child_link.clone());
                stack.push((child_link.clone(), new_path));
            }
        }
    }
    None
}

/// Construct entity path for a link by skipping every-other item in BFS chain
fn link_entity_path(
    adjacency: &HashMap<String, Vec<(Joint, String)>>,
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

// ----------------------------------------------------------------------------
// Loading geometry + materials

/// Load .stl into a Mesh3D
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
    // If <color> is present
    if let Some(c) = &mat.color {
        let rgba = &*c.rgba; // [f64; 4]
        info.color_rgba = Some([
            rgba[0] as f32,
            rgba[1] as f32,
            rgba[2] as f32,
            rgba[3] as f32,
        ]);
    }
    // If <texture> is present
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
fn load_image_as_rerun_buffer(path: &Path) -> Result<rerun::components::ImageBuffer> {
    let img = image::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open {path:?}: {e}"))?;
    let rgba8 = img.to_rgba8().into_raw(); // Vec<u8>
    let data_blob: rerun::datatypes::Blob = rgba8.into();
    let image_buf = ImageBuffer(data_blob);
    Ok(image_buf)
}

// ----------------------------------------------------------------------------
// Logging each link's geometry: now with global transform

/// For each link.visual:
///  1) Retrieve that link's global transform from BFS map (`link_global_tf`).
///  2) Build local visual transform from <origin xyz rpy>.
///  3) final_tf = link_global_tf * local_visual_tf
///  4) apply final_tf to the mesh
///  5) log
fn log_link_with_global_transform(
    link: &Link,
    link_global_tf: [f32; 16],
    entity_path: &str,
    urdf_dir: &PathBuf,
    all_materials_map: &HashMap<String, &Material>,
    rec: &RecordingStream,
) -> Result<()> {
    let mut doc_text = format!("Hierarchical URDF Link: {}\n", link.name);

    // Inertial summary
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

    // Each visual
    for (i, vis) in link.visual.iter().enumerate() {
        doc_text.push_str(&format!(
            "    #{} origin xyz={:?}, rpy={:?}\n",
            i, vis.origin.xyz, vis.origin.rpy
        ));

        // (1) gather material info
        let mut mat_info = RrMaterialInfo::default();
        if let Some(vis_mat) = &vis.material {
            let mat_name = &vis_mat.name;
            // If color/texture is None, assume named reference
            if vis_mat.color.is_none() && vis_mat.texture.is_none() {
                if let Some(global_mat) = all_materials_map.get(mat_name) {
                    mat_info = parse_urdf_material(global_mat, urdf_dir);
                }
            } else {
                mat_info = parse_urdf_material(vis_mat, urdf_dir);
            }
        }

        // (2) build geometry
        let mesh_entity_path = format!("{}/visual_{}", entity_path, i);
        let (mut mesh3d, extra_txt) = match &vis.geometry {
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
                    .map(|p| Position3D::from([p.x, p.y, p.z]))
                    .collect();
                let tri_idxs: Vec<TriangleIndices> = raw_i
                    .iter()
                    .map(|[a,b,c]| TriangleIndices::from([*a,*b,*c]))
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
                    .map(|p| Position3D::from([p.x, p.y, p.z]))
                    .collect();
                let tri_idxs: Vec<TriangleIndices> = raw_i
                    .iter()
                    .map(|[a,b,c]| TriangleIndices::from([*a,*b,*c]))
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
                    .map(|p| Position3D::from([p.x, p.y, p.z]))
                    .collect();
                let tri_idxs: Vec<TriangleIndices> = raw_i
                    .iter()
                    .map(|[a,b,c]| TriangleIndices::from([*a,*b,*c]))
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

        // (3) Build the local visual transform
        let xyz = [vis.origin.xyz[0], vis.origin.xyz[1], vis.origin.xyz[2]];
        let rpy = [vis.origin.rpy[0], vis.origin.rpy[1], vis.origin.rpy[2]];
        let local_tf = build_4x4_from_xyz_rpy(xyz, rpy);

        // (4) final_tf = link_global_tf * local_tf
        let final_tf = mat4x4_mul(link_global_tf, local_tf);

        // (5) Bake transform
        apply_4x4_to_mesh3d(&mut mesh3d, final_tf);

        // (6) Apply color/texture
        if let Some(rgba) = mat_info.color_rgba {
            let col_u8 = float_rgba_to_u8(rgba);
            let n_verts = mesh3d.vertex_positions.len();
            let mut all_colors = Vec::with_capacity(n_verts);
            for _ in 0..n_verts {
                all_colors.push(col_u8);
            }
            mesh3d = mesh3d.with_vertex_colors(all_colors);
        }
        if let Some(tex_path) = &mat_info.texture_path {
            match load_image_as_rerun_buffer(tex_path) {
                Ok(img_buf) => {
                    let (w, h) = image::image_dimensions(tex_path).unwrap_or((1, 1));
                    let format = ImageFormat::rgba8([w, h]);
                    mesh3d = mesh3d.with_albedo_texture(format, img_buf);
                }
                Err(e) => eprintln!("Warning: failed to load texture {tex_path:?}: {e}"),
            }
        }

        // (7) Log final mesh
        println!("======================");
        println!("rerun_log");
        println!("entity_path = '{}'", mesh_entity_path);
        println!(" => geometry has {} vertices", mesh3d.vertex_positions.len());
        rec.log(mesh_entity_path.as_str(), &mesh3d)?;
    }

    // Summarize link in a TextDocument
    println!("======================");
    println!("rerun_log");
    println!("entity_path = '{}'", entity_path);
    println!("entity = rerun::TextDocument(...)");
    rec.log(entity_path, &TextDocument::new(doc_text))?;

    Ok(())
}

// ----------------------------------------------------------------------------
// Main entry point: parse URDF, BFS to build global link transforms, then log

pub fn parse_and_log_urdf_hierarchy(urdf_path: &str, rec: &RecordingStream) -> Result<()> {
    // Parse URDF
    let robot_model = urdf_rs::read_file(urdf_path)
        .map_err(|e| anyhow::anyhow!("Failed to parse URDF {urdf_path:?}: {e}"))?;

    // Build adjacency
    let adjacency = build_adjacency(&robot_model.joints);

    // Find root link
    let root_link_name = find_root_link_name(&robot_model.links, &robot_model.joints)
        .unwrap_or_else(|| "base".to_owned());

    // Build a map link_name -> [f32;16] for each link's global transform
    let link_global_tf_map = build_link_global_transforms(&adjacency, &root_link_name, &robot_model.joints);

    // Collect global named materials
    let mut all_materials_map = HashMap::new();
    for m in &robot_model.materials {
        all_materials_map.insert(m.name.clone(), m);
    }

    // (A) Just log the root as a “view coordinates” or something similar
    println!("======================");
    println!("rerun_log");
    println!("entity_path = '' (root path)");
    println!("entity = (Pretend) rr.ViewCoordinates.RIGHT_HAND_Z_UP");
    println!("timeless = true");

    // (B) BFS chain for each link => log the geometry with the accumulated transform
    let urdf_dir = Path::new(urdf_path)
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    // We’ll still gather link references in a map for direct access
    let mut link_map: HashMap<String, &Link> = HashMap::new();
    for l in &robot_model.links {
        link_map.insert(l.name.clone(), l);
    }

    // For each link in the URDF
    for link in &robot_model.links {
        let link_name = &link.name;
        let entity_path = link_entity_path(&adjacency, &root_link_name, link_name)
            .unwrap_or_else(|| link_name.to_owned());

        // Grab the link’s global transform from BFS
        let link_global_tf = link_global_tf_map
            .get(link_name)
            .cloned()
            .unwrap_or_else(|| {
                eprintln!("Warning: no global transform found for link {}", link_name);
                // default identity
                [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ]
            });

        // Log the link’s geometry with the fully accumulated transform
        log_link_with_global_transform(
            link,
            link_global_tf,
            &entity_path,
            &urdf_dir,
            &all_materials_map,
            rec,
        )?;
    }

    Ok(())
}
