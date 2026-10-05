import bpy
import os
import math
import random
import traceback
import time
from mathutils import Vector, Matrix

# === CONFIG ===
ROOT_DIR = r"F:\Dataset"  # Your root dataset folder
RENDER_RES_X = 1920
RENDER_RES_Y = 1080
NUM_VIEWS = 6
TIMEOUT_SECONDS = 120  # 2 minutes timeout
LOG_FILE = os.path.join(ROOT_DIR, "render_progress_log.txt")
CYCLES_SAMPLES = 256  # Balanced quality/speed
HDRI_STRENGTH = 1.2

print("\n🚀 Smart Dataset Renderer (Textured + Untextured Models)\n")

# === LOGGING ===
def log_message(message):
    """Write to log file and console"""
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            t = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{t}] {message}\n")
        print(message)
    except Exception as e:
        print(f"⚠️ Log write failed: {e}")

def initialize_log(total_files):
    """Initialize log with header"""
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SMART BLENDER DATASET RENDERER LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Root Directory: {ROOT_DIR}\n")
            f.write(f"Resolution: {RENDER_RES_X}x{RENDER_RES_Y}\n")
            f.write(f"Samples: {CYCLES_SAMPLES}\n")
            f.write(f"Views per model: {NUM_VIEWS}\n")
            f.write(f"Timeout: {TIMEOUT_SECONDS}s (2 minutes)\n")
            f.write(f"Total GLB files found: {total_files}\n")
            f.write("="*80 + "\n\n")
        print(f"📝 Log initialized: {LOG_FILE}\n")
    except Exception as e:
        print(f"⚠️ Log initialization failed: {e}\n")

def update_progress(processed, total, success, skipped, timeout, failed):
    """Update progress in log"""
    remaining = total - processed
    percentage = (processed / total * 100) if total > 0 else 0
    
    msg = f"\n{'='*60}\n"
    msg += f"PROGRESS UPDATE\n"
    msg += f"{'='*60}\n"
    msg += f"Total files: {total}\n"
    msg += f"Processed: {processed}/{total} ({percentage:.1f}%)\n"
    msg += f"Remaining: {remaining}\n"
    msg += f"✅ Success: {success}\n"
    msg += f"⏭️  Skipped: {skipped}\n"
    msg += f"⏰ Timeout: {timeout}\n"
    msg += f"❌ Failed: {failed}\n"
    msg += f"{'='*60}\n"
    
    log_message(msg)

# === GPU SETUP ===
def setup_gpu():
    """Configure GPU rendering with automatic fallback"""
    device_used = "CPU"
    
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        # Try to enable CUDA/OPTIX/HIP devices if present
        # If not present, fallback to CPU
        try:
            prefs.compute_device_type = "CUDA"
        except Exception:
            # some builds require other strings; ignore if fails
            pass
        prefs.get_devices()
        
        gpu_found = False
        for d in prefs.devices:
            # enable all GPUs (CUDA / OPTIX / HIP / METAL)
            if d.type in {"CUDA", "OPTIX", "METAL", "HIP"}:
                d.use = True
                gpu_found = True
                device_used = d.type
            else:
                d.use = False
        
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.device = "GPU" if gpu_found else "CPU"
        
        # Quality settings
        bpy.context.scene.cycles.samples = CYCLES_SAMPLES
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        
        # Realism settings
        bpy.context.scene.cycles.max_bounces = 12
        bpy.context.scene.cycles.diffuse_bounces = 4
        bpy.context.scene.cycles.glossy_bounces = 4
        bpy.context.scene.cycles.caustics_reflective = True
        bpy.context.scene.cycles.caustics_refractive = True
        
        if gpu_found:
            log_message(f"🖥️  Using GPU ({device_used})")
        else:
            log_message("⚠️  No GPU found, using CPU")
            
        return device_used
        
    except Exception as e:
        log_message(f"⚠️ GPU setup failed: {e}, using CPU")
        return "CPU"

# === TEXTURE CHECK ===
def has_textures():
    """Check if model has image textures"""
    for obj in bpy.data.objects:
        if obj.type == "MESH" and getattr(obj.data, "materials", None):
            for mat in obj.data.materials:
                if mat and mat.use_nodes and getattr(mat, "node_tree", None):
                    for node in mat.node_tree.nodes:
                        if getattr(node, "type", "") == "TEX_IMAGE" and getattr(node, "image", None) is not None:
                            return True
    return False

# === REALISTIC MATERIALS (For untextured models) ===
MATERIAL_PRESETS = {
    'metal_brushed': {'color': [(0.7, 0.75), (0.7, 0.75), (0.75, 0.9)], 'metallic': (0.95, 1.0), 'roughness': (0.15, 0.35), 'weight': 0.20},
    'metal_polished': {'color': [(0.78, 0.85), (0.78, 0.85), (0.8, 0.92)], 'metallic': (0.98, 1.0), 'roughness': (0.05, 0.15), 'weight': 0.12},
    'plastic_matte': {'color': [(0.3, 0.85), (0.3, 0.8), (0.35, 0.75)], 'metallic': (0.0, 0.0), 'roughness': (0.4, 0.7), 'weight': 0.18},
    'plastic_glossy': {'color': [(0.4, 0.9), (0.35, 0.85), (0.3, 0.8)], 'metallic': (0.0, 0.0), 'roughness': (0.1, 0.25), 'weight': 0.15},
    'ceramic': {'color': [(0.85, 0.98), (0.83, 0.96), (0.8, 0.94)], 'metallic': (0.0, 0.0), 'roughness': (0.2, 0.4), 'weight': 0.10},
    'fabric': {'color': [(0.25, 0.75), (0.25, 0.7), (0.3, 0.65)], 'metallic': (0.0, 0.0), 'roughness': (0.7, 0.95), 'weight': 0.08},
    'glass': {'color': [(0.95, 1.0), (0.97, 1.0), (0.98, 1.0)], 'metallic': (0.0, 0.0), 'roughness': (0.0, 0.05), 'transmission': (0.95, 1.0), 'weight': 0.07},
    'wood': {'color': [(0.3, 0.7), (0.2, 0.5), (0.15, 0.35)], 'metallic': (0.0, 0.0), 'roughness': (0.5, 0.8), 'weight': 0.10}
}

def assign_realistic_materials():
    """Assign PBR materials ONLY for untextured models"""
    mesh_count = 0
    
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        
        # Choose material type
        weights = [p['weight'] for p in MATERIAL_PRESETS.values()]
        preset_name = random.choices(list(MATERIAL_PRESETS.keys()), weights=weights)[0]
        preset = MATERIAL_PRESETS[preset_name]
        
        # Create material
        mat = bpy.data.materials.new(f"PBR_{preset_name}_{obj.name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        # Principled BSDF
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        
        # Realistic color
        color = tuple(random.uniform(preset['color'][i][0], preset['color'][i][1]) for i in range(3)) + (1.0,)
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = random.uniform(*preset['metallic'])
        bsdf.inputs["Roughness"].default_value = random.uniform(*preset['roughness'])
        
        # Glass transmission
        if 'transmission' in preset:
            try:
                if "Transmission Weight" in bsdf.inputs:
                    bsdf.inputs["Transmission Weight"].default_value = random.uniform(*preset['transmission'])
                elif "Transmission" in bsdf.inputs:
                    bsdf.inputs["Transmission"].default_value = random.uniform(*preset['transmission'])
                bsdf.inputs["IOR"].default_value = random.uniform(1.45, 1.52)
            except:
                pass
        
        # Output
        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        mesh_count += 1
    
    log_message(f"  🎨 Applied realistic materials to {mesh_count} meshes")

# === BACKGROUND PRESETS ===
BACKGROUND_PRESETS = {
    'studio_white': {
        'type': 'color',
        'color': (0.95, 0.95, 0.95, 1.0),
        'strength': 1.5
    },
    'studio_gray': {
        'type': 'color', 
        'color': (0.2, 0.2, 0.2, 1.0),
        'strength': 1.2
    },
    'studio_black': {
        'type': 'color',
        'color': (0.05, 0.05, 0.05, 1.0),
        'strength': 1.0
    },
    'gradient_warm': {
        'type': 'gradient',
        'colors': [(0.9, 0.8, 0.7, 1.0), (0.7, 0.6, 0.5, 1.0)],
        'strength': 1.3
    },
    'gradient_cool': {
        'type': 'gradient', 
        'colors': [(0.7, 0.8, 0.9, 1.0), (0.5, 0.6, 0.7, 1.0)],
        'strength': 1.3
    },
    'sky_clear': {
        'type': 'sky',
        'sky_type': 'HOSEK_WILKIE',
        'turbidity': 1.0,
        'ground_albedo': 0.9,
        'strength': 1.2
    },
    'sky_overcast': {
        'type': 'sky',
        'sky_type': 'HOSEK_WILKIE', 
        'turbidity': 4.0,
        'ground_albedo': 0.3,
        'strength': 0.8
    }
}

def setup_background():
    """Setup random background for each model"""
    preset_name = random.choice(list(BACKGROUND_PRESETS.keys()))
    preset = BACKGROUND_PRESETS[preset_name]
    
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    
    bg = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")
    
    if preset['type'] == 'color':
        # Solid color background
        bg.inputs["Color"].default_value = preset['color']
        bg.inputs["Strength"].default_value = preset['strength']
        links.new(bg.outputs["Background"], output.inputs["Surface"])
        
    elif preset['type'] == 'gradient':
        # Gradient background
        tex_coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        gradient = nodes.new("ShaderNodeValToRGB")
        
        # Create gradient colors
        gradient.color_ramp.elements[0].color = preset['colors'][0]
        gradient.color_ramp.elements[1].color = preset['colors'][1]
        
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], gradient.inputs["Fac"])
        links.new(gradient.outputs["Color"], bg.inputs["Color"])
        links.new(bg.outputs["Background"], output.inputs["Surface"])
        
        bg.inputs["Strength"].default_value = preset['strength']
        
    elif preset['type'] == 'sky':
        # Sky background
        sky = nodes.new("ShaderNodeTexSky")
        sky.sky_type = preset['sky_type']
        sky.turbidity = preset['turbidity']
        sky.ground_albedo = preset['ground_albedo']
        sky.sun_disc = False
        
        links.new(sky.outputs["Color"], bg.inputs["Color"])
        links.new(bg.outputs["Background"], output.inputs["Surface"])
        bg.inputs["Strength"].default_value = preset['strength']
    
    log_message(f"  🎭 Background: {preset_name}")

def setup_studio_lighting():
    """3-point studio lighting"""
    # Clear existing lights
    bpy.ops.object.select_all(action='DESELECT')
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()
    
    # Key light
    key_angle = random.uniform(30, 60)
    key_dist = random.uniform(6, 9)
    bpy.ops.object.light_add(type='AREA', location=(
        key_dist * math.cos(math.radians(key_angle)),
        key_dist * math.sin(math.radians(key_angle)),
        random.uniform(4, 6)
    ))
    key = bpy.context.object
    key.data.energy = random.uniform(1000, 1500)
    key.data.size = random.uniform(3, 5)
    
    # Fill light
    fill_angle = key_angle + 180
    bpy.ops.object.light_add(type='AREA', location=(
        5 * math.cos(math.radians(fill_angle)),
        5 * math.sin(math.radians(fill_angle)),
        3
    ))
    fill = bpy.context.object
    fill.data.energy = random.uniform(400, 700)
    fill.data.size = 3
    
    # Rim light
    bpy.ops.object.light_add(type='SPOT', location=(0, 5, 6))
    rim = bpy.context.object
    rim.data.energy = random.uniform(800, 1200)

# === MODEL SCALING AND CENTERING ===
def calculate_bounding_box_world():
    """Calculate the world-space axis-aligned bounding box of all mesh objects"""
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if not meshes:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    
    # Initialize with extreme values
    min_coord = Vector((float('inf'), float('inf'), float('inf')))
    max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    for obj in meshes:
        # Use obj.matrix_world to transform vertices to world coordinates
        mat = obj.matrix_world
        # ensure mesh has vertices
        if not getattr(obj.data, "vertices", None):
            continue
        for v in obj.data.vertices:
            world_v = mat @ v.co
            min_coord.x = min(min_coord.x, world_v.x)
            min_coord.y = min(min_coord.y, world_v.y)
            min_coord.z = min(min_coord.z, world_v.z)
            max_coord.x = max(max_coord.x, world_v.x)
            max_coord.y = max(max_coord.y, world_v.y)
            max_coord.z = max(max_coord.z, world_v.z)
    
    return min_coord, max_coord

def camera_vertical_fov(scene, cam_obj):
    """Compute the camera's vertical FOV in radians using lens and sensor size,
       taking into account sensor_fit and render aspect."""
    cam = cam_obj.data
    # focal length (mm)
    f = cam.lens
    # sensor size
    if cam.sensor_fit == 'VERTICAL':
        sensor = cam.sensor_height
    elif cam.sensor_fit == 'HORIZONTAL':
        sensor = cam.sensor_width
    else:
        # AUTO: choose based on aspect ratio
        aspect = scene.render.resolution_x / scene.render.resolution_y
        if aspect >= 1.0:
            sensor = cam.sensor_width
            # convert horizontal FOV to vertical based on aspect
            fov_h = 2.0 * math.atan(sensor / (2.0 * f))
            # vertical fov:
            fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) * (1.0 / aspect))
            return fov_v
        else:
            sensor = cam.sensor_height
    fov = 2.0 * math.atan(sensor / (2.0 * f))
    return fov

def center_and_scale_model():
    """Center the model at origin and scale to fit camera view.
       Returns optimal camera distance in world units."""
    min_coord, max_coord = calculate_bounding_box_world()
    center = (min_coord + max_coord) / 2.0
    dims = max_coord - min_coord
    max_dim = max(dims.x, dims.y, dims.z)
    
    if max_dim == 0:
        return 5.0  # fallback distance if model is degenerate
    
    # Target size (we want the largest dimension to be close to target_size units)
    target_size = 2.0
    scale_factor = target_size / max_dim
    
    # Apply world-space transform: translate by -center, then scale about origin
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    for obj in meshes:
        # Compute new matrix: S * T * old_matrix
        T = Matrix.Translation(-center)
        S = Matrix.Scale(scale_factor, 4)
        obj.matrix_world = S @ T @ obj.matrix_world
    
    # Recalculate bbox after transform
    min_c, max_c = calculate_bounding_box_world()
    dims_after = max_c - min_c
    max_dim_after = max(dims_after.x, dims_after.y, dims_after.z)
    
    # Choose camera and compute vertical FOV
    # Create a temporary camera to compute FOV if scene has none
    scene = bpy.context.scene
    if not scene.camera:
        bpy.ops.object.camera_add(location=(0, 0, max_dim_after * 2.0))
        temp_cam = bpy.context.object
        scene.camera = temp_cam
        created_temp_cam = True
    else:
        temp_cam = scene.camera
        created_temp_cam = False
    
    temp_cam.data.lens = 50  # set focal length for consistent calculation
    # ensure sensor values exist
    if temp_cam.data.sensor_width is None:
        temp_cam.data.sensor_width = 36.0
    if temp_cam.data.sensor_height is None:
        temp_cam.data.sensor_height = 24.0
    
    fov_v = camera_vertical_fov(scene, temp_cam)
    
    # padding to ensure model fits comfortably
    padding = 1.2
    # The required distance so the half-height fits in half-FOV: distance = (half_height) / tan(fov_v/2)
    half_height = max_dim_after / 2.0
    optimal_distance = (half_height / math.tan(fov_v / 2.0)) * padding
    optimal_distance = max(optimal_distance, max_dim_after * 1.2)
    
    # If we created a temp camera, remove it (we don't want to keep extra cameras)
    if created_temp_cam:
        bpy.data.objects.remove(temp_cam, do_unlink=True)
        scene.camera = None
    
    log_message(f"📐 Original bbox dims: {dims.x:.3f}, {dims.y:.3f}, {dims.z:.3f}")
    log_message(f"📐 Scaled bbox dims: {dims_after.x:.3f}, {dims_after.y:.3f}, {dims_after.z:.3f}")
    log_message(f"📐 Scale factor applied: {scale_factor:.5f}")
    log_message(f"📐 Calculated camera distance: {optimal_distance:.3f}")
    
    return optimal_distance

# === ARBITRARY CAMERA POSITIONS ===
def generate_arbitrary_camera_positions(distance):
    """Generate truly arbitrary camera positions with varied angles and distances"""
    positions = []
    for i in range(NUM_VIEWS):
        azimuth = random.uniform(0, 360)
        elevation = random.uniform(10, 80)
        cam_distance = distance * random.uniform(0.8, 1.2)
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        x = cam_distance * math.cos(el_rad) * math.cos(az_rad)
        y = cam_distance * math.cos(el_rad) * math.sin(az_rad)
        z = cam_distance * math.sin(el_rad)
        x += random.uniform(-0.2, 0.2)
        y += random.uniform(-0.2, 0.2)
        z += random.uniform(-0.1, 0.1)
        positions.append({
            'loc': (x, y, z),
            'name': f"r_{i}",
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': cam_distance
        })
    return positions

# === RENDER FUNCTION WITH TIMEOUT ===
def render_glb_with_timeout(glb_path, device_used):
    """Render GLB with 2-minute timeout protection"""
    name = os.path.basename(glb_path)
    folder = os.path.dirname(glb_path)
    img_dir = os.path.join(folder, "images")
    
    log_message(f"\n{'='*60}\n🧩 Processing: {name}\n{'='*60}")
    
    start_time = time.time()
    
    # Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_gpu()
    
    # Import GLB
    try:
        bpy.ops.import_scene.gltf(filepath=glb_path)
        log_message(f"✅ GLB imported")
    except Exception as e:
        log_message(f"❌ Import failed: {e}")
        return 'IMPORT_FAIL', 0
    
    # Check if model has textures
    has_tex = has_textures()
    
    if has_tex:
        log_message(f"🎨 Model has textures - rendering with ORIGINAL textures")
    else:
        log_message(f"🎨 No textures found - applying realistic PBR materials")
        assign_realistic_materials()
    
    # Setup random background and lighting
    setup_background()
    setup_studio_lighting()
    
    # Center and scale model to fit camera, get optimal distance
    optimal_dist = center_and_scale_model()
    
    # Render settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.resolution_x = RENDER_RES_X
    scene.render.resolution_y = RENDER_RES_Y
    scene.render.film_transparent = False
    
    # Setup camera (create new camera each model to avoid side effects)
    bpy.ops.object.camera_add(location=(optimal_dist, 0, 0))
    cam = bpy.context.object
    cam.data.lens = 50
    # Choose sensor_fit automatically based on aspect ratio
    aspect = scene.render.resolution_x / scene.render.resolution_y
    cam.data.sensor_fit = 'AUTO'
    scene.camera = cam
    
    # Target at origin to track
    target = bpy.data.objects.new("Target", None)
    scene.collection.objects.link(target)
    target.location = (0, 0, 0)
    
    # Check if images folder exists
    os.makedirs(img_dir, exist_ok=True)
    
    # Render each view
    render_count = 0
    positions = generate_arbitrary_camera_positions(optimal_dist)
    
    for pos in positions:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT_SECONDS:
            log_message(f"⏰ TIMEOUT at {elapsed:.1f}s - stopping renders")
            try:
                bpy.data.objects.remove(target)
            except Exception:
                pass
            return 'TIMEOUT', render_count
        
        # Position camera
        cam.location = pos['loc']
        
        # Remove existing TRACK_TO constraints
        for c in list(cam.constraints):
            cam.constraints.remove(c)
        track = cam.constraints.new(type='TRACK_TO')
        track.target = target
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'
        
        # Render
        out_path = os.path.join(img_dir, f"{pos['name']}.png")
        scene.render.filepath = out_path
        
        log_message(f"📸 Rendering {pos['name']} (Az:{pos['azimuth']:.0f}°, El:{pos['elevation']:.0f}°, Dist:{pos['distance']:.1f})")
        
        try:
            bpy.ops.render.render(write_still=True)
            log_message(f"   ✅ Saved: {pos['name']}.png")
            render_count += 1
        except Exception as e:
            log_message(f"   ❌ Render failed: {e}")
    
    # Cleanup
    try:
        bpy.data.objects.remove(target)
    except Exception:
        pass
    
    total_time = time.time() - start_time
    log_message(f"✅ Completed {render_count}/{NUM_VIEWS} renders in {total_time:.1f}s")
    
    return 'SUCCESS', render_count

# === MAIN LOOP ===
log_message("🔍 Scanning for GLB files...")

# Find all GLB files
glb_files = []
for root, _, files in os.walk(ROOT_DIR):
    for f in files:
        if f.lower().endswith(".glb"):
            glb_files.append(os.path.join(root, f))

total_files = len(glb_files)
log_message(f"📂 Found {total_files} GLB files\n")

initialize_log(total_files)
device_used = setup_gpu()

# Counters
processed = 0
success = 0
timeout_count = 0
skipped = 0
failed = 0

# Process each file
for idx, glb_path in enumerate(glb_files, 1):
    try:
        log_message(f"\n[{idx}/{total_files}] Starting: {os.path.basename(glb_path)}")
        
        status, render_count = render_glb_with_timeout(glb_path, device_used)
        
        processed += 1
        
        if status == 'SUCCESS':
            success += 1
        elif status == 'TIMEOUT':
            timeout_count += 1
        elif status == 'IMPORT_FAIL':
            failed += 1
        else:
            skipped += 1
        
        # Progress update every 10 files
        if processed % 10 == 0:
            update_progress(processed, total_files, success, skipped, timeout_count, failed)
            
    except Exception as e:
        failed += 1
        log_message(f"❌ Critical error on {os.path.basename(glb_path)}: {str(e)}")
        log_message(traceback.format_exc())

# === FINAL SUMMARY ===
log_message("\n" + "="*80)
log_message("                    FINAL SUMMARY")
log_message("="*80)
log_message(f"🎬 Total GLB files: {total_files}")
log_message(f"✅ Successfully rendered: {success}")
log_message(f"⏰ Timeout (>2min): {timeout_count}")
log_message(f"⚠️  Skipped: {skipped}")
log_message(f"❌ Failed: {failed}")
log_message(f"🖼️  Views per model: {NUM_VIEWS}")
log_message(f"📁 Output: [model_folder]/images/r_0.png to r_5.png")
log_message(f"📝 Full log: {LOG_FILE}")
log_message("="*80)
log_message(f"✅ Rendering completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
log_message("="*80 + "\n")

print("\n🎉 All renders completed! Check log for details.\n")
