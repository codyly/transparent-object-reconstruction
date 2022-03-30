

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#  include <GL/wglew.h>
#  include <GL/freeglut.h>
#  else
#  include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <sutil.h>
#include "common.h"
#include "random.h"
#include <Arcball.h>
#include <OptiXMesh.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <math.h>
//#include <math_functions.h>
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "myProject";

static float rand_range(float min, float max)
{
	static unsigned int seed = 0u;
	return min + (max - min) * rnd(seed);
}


//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context      context;
uint32_t     width = 800u;
uint32_t     height = 800u;
bool         use_pbo = true;
bool         use_tri_api = true;
bool         ignore_mats = false;
optix::Aabb    aabb;
bool		 rotation = false;
float3 meshmove;
static int t = -1;

std::string  texture_path;
const char* tutorial_ptx;
int          tutorial_number = 10;

// Camera state
float3       camera_up;
float3       camera_lookat;
float3       camera_eye;
Matrix4x4    camera_rotate;
sutil::Arcball arcball;

// Mouse state
optix::int2       mouse_prev_pos;
int        mouse_button;

//instance
std::vector<GeometryInstance> gis;
GeometryGroup geometrygroup;
GeometryGroup gg;
Group root_group;
Transform transform;
//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void createGeometry();
void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize(int* argc, char** argv);
void glutRun();

void glutDisplay();
void glutKeyboardPress(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutResize(int w, int h);


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer()
{
	return context["output_buffer"]->getBuffer();
}


void destroyContext()
{
	if (context)
	{
		context->destroy();
		context = 0;
	}
}


void registerExitHandler()
{
	// register shutdown handler
#ifdef _WIN32
	glutCloseFunc(destroyContext);  // this function is freeglut-only
#else
	atexit(destroyContext);
#endif
}


void createContext()
{
	// Set up context
	context = Context::create();
	context->setRayTypeCount(2);
	context->setEntryPointCount(1);
	context->setStackSize(1080);
	context->setMaxTraceDepth(31);

	// Note: high max depth for reflection and refraction through glass
	context["max_depth"]->setInt(1000);
	context["scene_epsilon"]->setFloat(1.e-4f);
	context["importance_cutoff"]->setFloat(0.01f);
	context["ambient_light_color"]->setFloat(0.31f, 0.31f, 0.31f);

	// Output buffer
	// First allocate the memory for the GL buffer, then attach it to OptiX.
	GLuint vbo = 0;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, 4 * width * height, NULL, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo);
	context["output_buffer"]->set(buffer);

	// Ray generation program
	const std::string camera_name = "pinhole_camera";
	Program ray_gen_program = context->createProgramFromPTXString(tutorial_ptx, camera_name);
	context->setRayGenerationProgram(0, ray_gen_program);

	// Exception program
	Program exception_program = context->createProgramFromPTXString(tutorial_ptx, "exception");
	context->setExceptionProgram(0, exception_program);
	context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

	// Miss program
	const std::string miss_name = "envmap_miss";
	context->setMissProgram(0, context->createProgramFromPTXString(tutorial_ptx, miss_name));
	const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
	const std::string texpath = texture_path + "/" + std::string("CedarCit.hdr");
	context["envmap"]->setTextureSampler(sutil::loadTexture(context, texpath, default_color));
	context["bg_color"]->setFloat(make_float3(0.34f, 0.55f, 0.85f));

	// 3D solid noise buffer, 1 float channel, all entries in the range [0.0, 1.0].

	const int tex_width = 64;
	const int tex_height = 64;
	const int tex_depth = 64;
	Buffer noiseBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, tex_width, tex_height, tex_depth);
	float* tex_data = (float*)noiseBuffer->map();

	// Random noise in range [0, 1]
	for (int i = tex_width * tex_height * tex_depth; i > 0; i--) {
		// One channel 3D noise in [0.0, 1.0] range.
		*tex_data++ = rand_range(0.0f, 1.0f);
	}
	noiseBuffer->unmap();


	// Noise texture sampler
	TextureSampler noiseSampler = context->createTextureSampler();

	noiseSampler->setWrapMode(0, RT_WRAP_REPEAT);
	noiseSampler->setWrapMode(1, RT_WRAP_REPEAT);
	noiseSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	noiseSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	noiseSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
	noiseSampler->setMaxAnisotropy(1.0f);
	noiseSampler->setMipLevelCount(1);
	noiseSampler->setArraySize(1);
	noiseSampler->setBuffer(0, 0, noiseBuffer);

	context["noise_texture"]->setTextureSampler(noiseSampler);

	const std::string tagtexpath = texture_path + "/" + std::string("board.ppm");
	context["tagsmap"]->setTextureSampler(sutil::loadTexture(context, tagtexpath, default_color));
}

float4 make_plane(float3 n, float3 p)
{
	n = normalize(n);
	float d = -dot(n, p);
	return make_float4(n, d);
}

Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;

GeometryInstance createParallelogram(
	const float3& anchor,
	const float3& offset1,
	const float3& offset2)
{
	Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	parallelogram->setIntersectionProgram(pgram_intersection);
	parallelogram->setBoundingBoxProgram(pgram_bounding_box);

	float3 normal = normalize(cross(offset1, offset2));
	float d = dot(normal, anchor);
	float4 plane = make_float4(normal, d);

	float3 v1 = offset1 / dot(offset1, offset1);
	float3 v2 = offset2 / dot(offset2, offset2);

	parallelogram["plane"]->setFloat(plane);
	parallelogram["anchor"]->setFloat(anchor);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);

	GeometryInstance gi = context->createGeometryInstance();
	gi->setGeometry(parallelogram);
	return gi;
}

void setMaterial(
	GeometryInstance& gi,
	Material material,
	const std::string& color_name,
	const float3& color)
{
	gi->addMaterial(material);
	gi[color_name]->setFloat(color);
}


float angle_offset = 0.0f;

void createGeometry()
{
	const float3 white = make_float3(0.8f, 0.8f, 0.8f);

	const float3 light_em = make_float3(15.0f, 15.0f, 5.0f);
	// Create box
	const char* ptx = sutil::getPtxString(SAMPLE_NAME, "box.cu");
	Program box_bounds = context->createProgramFromPTXString(ptx, "box_bounds");
	Program box_intersect = context->createProgramFromPTXString(ptx, "box_intersect");
	Geometry box = context->createGeometry();
	box->setPrimitiveCount(1u);
	box->setBoundingBoxProgram(box_bounds);
	box->setIntersectionProgram(box_intersect);
	/*box["boxmin"]->setFloat(-60.0f, 0.0f, -100.0f);
	box["boxmax"]->setFloat(100.0f, 100.0f, 100.0f);*/
	box["boxmin"]->setFloat(-100.0f, -300.0f, -300.0f);
	box["boxmax"]->setFloat(500.0f, 300.0f, 300.0f);

	Geometry screen = context->createGeometry();
	screen->setPrimitiveCount(1u);
	screen->setBoundingBoxProgram(box_bounds);
	screen->setIntersectionProgram(box_intersect);
	screen["boxmin"]->setFloat(-15.0f, 40.0f, -15.0f);
	screen["boxmax"]->setFloat(15.0f, 60.0f, 15.0f);

	// inside cube
	Geometry cube = context->createGeometry();
	cube->setPrimitiveCount(1u);
	cube->setBoundingBoxProgram(box_bounds);
	cube->setIntersectionProgram(box_intersect);
	cube["boxmin"]->setFloat(-1.0f, 48.5f, -1.0f);
	cube["boxmax"]->setFloat(1.0f, 52.0f, 1.0f);

	// tag board
	Geometry tagboard = context->createGeometry();
	tagboard->setPrimitiveCount(1u);
	tagboard->setBoundingBoxProgram(box_bounds);
	tagboard->setIntersectionProgram(box_intersect);
	tagboard["boxmin"]->setFloat(-1.0f, -80.0f, -80.0f);
	tagboard["boxmax"]->setFloat(1.0f, 80.0f, 80.0f);

	const char* para_ptx = sutil::getPtxString(SAMPLE_NAME, "parallelogram.cu");
	pgram_bounding_box = context->createProgramFromPTXString(para_ptx, "bounds");
	pgram_intersection = context->createProgramFromPTXString(para_ptx, "intersect");


	// sphere
	const char* sphere_ptx = sutil::getPtxString(SAMPLE_NAME, "sphere.cu");
	Program sphere_bounds = context->createProgramFromPTXString(sphere_ptx, "bounds");
	Program sphere_intersect = context->createProgramFromPTXString(sphere_ptx, "intersect");
	Geometry sphere = context->createGeometry();
	sphere->setPrimitiveCount(1u);
	sphere->setBoundingBoxProgram(sphere_bounds);
	sphere->setIntersectionProgram(sphere_intersect);
	sphere["sphere"]->setFloat(0, 50.0f, 0, 2.0f);

	// cylinder
	const char* cylinder_ptx = sutil::getPtxString(SAMPLE_NAME, "cylinder.cu");
	Program cylinder_bounds = context->createProgramFromPTXString(cylinder_ptx, "cylinder_bounds");
	Program cylinder_intersect = context->createProgramFromPTXString(cylinder_ptx, "cylinder_intersect");
	Geometry cylinder = context->createGeometry();
	cylinder->setPrimitiveCount(1u);
	cylinder->setBoundingBoxProgram(cylinder_bounds);
	cylinder->setIntersectionProgram(cylinder_intersect);
	/*cylinder["cylinder_boxmin"]->setFloat(-10.0f, 30.0f, -10.0f);
	cylinder["cylinder_boxmax"]->setFloat(10.0f, 70.0f, 10.0f);*/
	cylinder["cylinder_boxmin"]->setFloat(-80.0f, -80.0f, -80.0f);
	cylinder["cylinder_boxmax"]->setFloat(80.0f, 80.0f, 80.0f);

	// Create chull
	Geometry chull = 0;
	if (tutorial_number >= 9) {
		const char* chull_ptx = sutil::getPtxString(SAMPLE_NAME, "chull.cu");
		chull = context->createGeometry();
		chull->setPrimitiveCount(1u);
		chull->setBoundingBoxProgram(context->createProgramFromPTXString(chull_ptx, "chull_bounds"));
		chull->setIntersectionProgram(context->createProgramFromPTXString(chull_ptx, "chull_intersect"));
		Buffer plane_buffer = context->createBuffer(RT_BUFFER_INPUT);
		plane_buffer->setFormat(RT_FORMAT_FLOAT4);
		int nsides = 6;
		plane_buffer->setSize(nsides + 2);
		float4* chplane = (float4*)plane_buffer->map();
		float radius = 1.5f;
		float3 xlate = make_float3(0.0f, 47.0f, 0.0f);
		angle_offset = M_PIf / 3.0f;
		for (int i = 0; i < nsides; i++) {
			float angle = float(i) / float(nsides) * M_PIf * 2.0f;
			float x = cos(angle + angle_offset);
			float y = sin(angle + angle_offset);
			chplane[i] = make_plane(make_float3(x, 0, y), make_float3(x * radius, 0, y * radius) + xlate);
		}
		float min = 0.02f;
		float max = 1.5f;
		chplane[nsides + 0] = make_plane(make_float3(0, -1, 0), make_float3(0, min, 0) + xlate);
		float angle = 5.f / nsides * M_PIf * 2;
		float pitch = 0.7f;
		float ytopOffset = (radius / pitch) / cos(M_PIf / nsides);
		chplane[nsides + 1] = make_plane(make_float3(cos(angle + angle_offset), pitch, sin(angle + angle_offset)), make_float3(0, max, 0) + xlate);
		//chplane[nsides + 1] = make_plane(make_float3(0, 1, 0), make_float3(0, max, 0) + xlate);
		plane_buffer->unmap();
		chull["planes"]->setBuffer(plane_buffer);
		float radoffset = radius / cos(M_PIf / nsides);
		chull["chull_bbmin"]->setFloat(-radoffset + xlate.x, min + xlate.y, -radoffset + xlate.z);
		chull["chull_bbmax"]->setFloat(radoffset + xlate.x, max + xlate.y + ytopOffset, radoffset + xlate.z);
	}

	// Materials
	// Box Material
	std::string box_chname;
	box_chname = "box_closest_hit_radiance";
	box_chname = "closest_hit_radiance3";

	Material box_matl = context->createMaterial();
	Program box_ch = context->createProgramFromPTXString(tutorial_ptx, box_chname.c_str());
	box_matl->setClosestHitProgram(0, box_ch);
	Program box_ah = context->createProgramFromPTXString(tutorial_ptx, "any_hit_shadow");
	//box_matl->setAnyHitProgram(1, box_ah);
	box_matl["Ka"]->setFloat(0.1f, 0.1f, 0.1f);
	box_matl["Kd"]->setFloat(0.9f, 0.9f, 0.9f);
	box_matl["Ks"]->setFloat(0.05f, 0.05f, 0.05f);
	box_matl["phong_exp"]->setFloat(10);
	box_matl["reflectivity_n"]->setFloat(0.2f, 0.2f, 0.2f);
	box_matl["metalroughness"]->setFloat(5);

	std::string screen_chname;
	screen_chname = "closest_hit_radiance3";
	Material screen_matl = context->createMaterial();
	Program screen_ch = context->createProgramFromPTXString(tutorial_ptx, screen_chname.c_str());
	screen_matl->setClosestHitProgram(0, screen_ch);
	//screen_matl->setAnyHitProgram(1, box_ah);
	screen_matl["Ka"]->setFloat(3.0f, 3.0f, 3.0f);
	screen_matl["Kd"]->setFloat(0.6f, 0.7f, 0.8f);
	screen_matl["Ks"]->setFloat(0.8f, 0.9f, 0.8f);

	std::string tag_chname;
	tag_chname = "closest_hit_tags";
	Material tag_matl = context->createMaterial();
	Program tag_ch = context->createProgramFromPTXString(tutorial_ptx, tag_chname.c_str());
	tag_matl->setClosestHitProgram(0, tag_ch);
	Program tag_ah = context->createProgramFromPTXString(tutorial_ptx, "any_hit_shadow");
	//tag_matl->setAnyHitProgram(1, tag_ah);
	tag_matl["Ka"]->setFloat(0.3f, 0.3f, 0.3f);
	tag_matl["Kd"]->setFloat(0.6f, 0.7f, 0.8f);
	tag_matl["Ks"]->setFloat(0.8f, 0.9f, 0.8f);
	tag_matl["phong_exp"]->setFloat(100);
	tag_matl["reflectivity_n"]->setFloat(0.0f, 0.0f, 0.0f);
	tag_matl["metalroughness"]->setFloat(5);


	// Glass material

	// Transparent object material 
	Material glass_matl;
	if (chull.get() || cube.get()) {
		Program glass_ch = context->createProgramFromPTXString(tutorial_ptx, "glass_closest_hit_radiance");
		const std::string glass_ahname = tutorial_number >= 10 ? "glass_any_hit_shadow" : "any_hit_shadow";
		Program glass_ah = context->createProgramFromPTXString(tutorial_ptx, glass_ahname.c_str());
		glass_matl = context->createMaterial();
		glass_matl->setClosestHitProgram(0, glass_ch);
		//glass_matl->setAnyHitProgram(1, glass_ah);
		glass_matl["importance_cutoff"]->setFloat(1e-3f);
		glass_matl["cutoff_color"]->setFloat(1.0f, 1.0f, 1.0f);
		glass_matl["fresnel_exponent"]->setFloat(3.0f);
		glass_matl["fresnel_minimum"]->setFloat(0.1f);
		glass_matl["fresnel_maximum"]->setFloat(0.1f);
		glass_matl["refraction_index"]->setFloat(1.0f);
		glass_matl["refraction_color"]->setFloat(0.0f, 1.0f, 0.0f);
		glass_matl["reflection_color"]->setFloat(1.0f, 1.0f, 1.0f);
		glass_matl["refraction_maxdepth"]->setInt(500);
		glass_matl["reflection_maxdepth"]->setInt(0);
		float3 extinction = make_float3(.85f, .94f, .80f);
		glass_matl["extinction_constant"]->setFloat(log(extinction.x), log(extinction.y), log(extinction.z));
		glass_matl["shadow_attenuation"]->setFloat(0.4f, 0.7f, 0.4f);
	}
	Material cube_matl = glass_matl;
	Material cylinder_matl = glass_matl;
	Material sphere_matl = glass_matl;
	Material tagboard_matl = tag_matl;
	//screen_matl = glass_matl;

	// Container material
	Material glass_matl2;
	if (cylinder.get()) {
		Program glass_ch2 = context->createProgramFromPTXString(tutorial_ptx, "glass_closest_hit_radiance");
		const std::string glass_ahname2 = tutorial_number >= 10 ? "glass_any_hit_shadow" : "any_hit_shadow";
		Program glass_ah2 = context->createProgramFromPTXString(tutorial_ptx, glass_ahname2.c_str());
		glass_matl2 = context->createMaterial();
		glass_matl2->setClosestHitProgram(0, glass_ch2);
		//glass_matl2->setAnyHitProgram(1, glass_ah2);
		glass_matl2["importance_cutoff"]->setFloat(1e-3f);
		glass_matl2["cutoff_color"]->setFloat(1.0f, 1.0, 1.0f);
		glass_matl2["fresnel_exponent"]->setFloat(3.0f);
		glass_matl2["fresnel_minimum"]->setFloat(0.1f);
		glass_matl2["fresnel_maximum"]->setFloat(1.0f);
		glass_matl2["refraction_index"]->setFloat(1.5f);
		glass_matl2["refraction_color"]->setFloat(1.0f, 1.0f, 1.0f);
		glass_matl2["reflection_color"]->setFloat(1.0f, 1.0f, 1.0f);
		glass_matl2["refraction_maxdepth"]->setInt(50);
		glass_matl2["reflection_maxdepth"]->setInt(0);
		float3 extinction = make_float3(1.0f, 1.0f, 1.0f);
		glass_matl2["extinction_constant"]->setFloat(log(extinction.x), log(extinction.y), log(extinction.z));
		glass_matl2["shadow_attenuation"]->setFloat(1.0f, 1.0f, 1.0f);
	}
	// cube_matl = glass_matl2;
	// Create GIs for each piece of geometry
	gis.push_back(context->createGeometryInstance(box, &box_matl, &box_matl + 1));
	//gis.push_back(context->createGeometryInstance(parallelogram, &floor_matl, &floor_matl + 1));

	if (cylinder.get())
		gis.push_back(context->createGeometryInstance(cylinder, &glass_matl2, &glass_matl2 + 1));
	if (chull.get())
		gis.push_back(context->createGeometryInstance(chull, &glass_matl, &glass_matl + 1));
	if (cube.get())
		gis.push_back(context->createGeometryInstance(cube, &cube_matl, &cube_matl + 1));
	if (sphere.get())
		gis.push_back(context->createGeometryInstance(sphere, &sphere_matl, &sphere_matl + 1));
	if (tagboard.get())
		gis.push_back(context->createGeometryInstance(tagboard, &tagboard_matl, &tagboard_matl + 1));
	if (screen.get())
		gis.push_back(context->createGeometryInstance(screen, &screen_matl, &screen_matl + 1));

	


	// Place all in group
	geometrygroup = context->createGeometryGroup();
	geometrygroup->setChildCount(2);
	geometrygroup->setChild(0, gis[0]);

	if (cylinder.get()) {
		geometrygroup->setChild(1, gis[1]);
	}
	//geometrygroup->setChild(1, gis[5]);
	//geometrygroup->setChild(1, gis[6]);

	geometrygroup->setAcceleration(context->createAcceleration("Trbvh"));

	gg = context->createGeometryGroup();
	gg->setChildCount(1);
	//gg->setChild(0, gis[3]);
	//gg->setChild(1, gis[2]);
	//gg->setChild(0, gis[4]);

	OptiXMesh mesh;
	mesh.context = context;
	mesh.use_tri_api = use_tri_api;
	mesh.ignore_mats = ignore_mats;
	mesh.material = glass_matl;
	
	std::string filename = std::string(sutil::samplesDir()) + "/data/a1.obj";
	loadMesh(filename, mesh);

	meshmove = -(mesh.bbox_max + mesh.bbox_min) / 2;



	aabb.set(mesh.bbox_min, mesh.bbox_max);
	//std::cout << (mesh.bbox_max + mesh.bbox_min)/2 << std::endl;
	gg->setChild(0, mesh.geom_instance);
	gg->setAcceleration(context->createAcceleration("Trbvh"));

	

	
	//geometrygroup->setAcceleration( context->createAcceleration("NoAccel") );

	root_group = context->createGroup();
	root_group->setAcceleration(context->createAcceleration("Trbvh"));
	root_group->addChild(geometrygroup);
	

	float m[16] = { 1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
					meshmove.x, meshmove.y, meshmove.z, 1};

	Matrix4x4 mat(m);
	transform = context->createTransform();
	transform->setChild(gg);
	transform->setMatrix(true, mat.getData(), mat.inverse().getData());
	transform->validate();



	root_group->addChild(transform);
	root_group->getAcceleration()->markDirty();

	context["top_object"]->set(root_group);
	context["top_shadower"]->set(root_group);
}


void setupCamera()
{
	camera_eye = make_float3(250.0f, 0.0f, -0.0f);
	camera_lookat = make_float3(0.0f, 0.0f, 0.0f);
	camera_up = make_float3(0.0f, 1.0f, 0.0f);

	camera_rotate = Matrix4x4::identity();
}


void setupLights()
{
	BasicLight lights[] = {
		{ make_float3(20.0f, 100.0f, -20.0f), make_float3(0.9f,0.9f,0.9f), 1 },
	    { make_float3(20.0f, -100.0f, 20.0f), make_float3(0.9f,0.9f,0.9f), 1 },
	};

	Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(BasicLight));
	light_buffer->setSize(sizeof(lights) / sizeof(lights[0]));
	memcpy(light_buffer->map(), lights, sizeof(lights));
	light_buffer->unmap();

	context["lights"]->set(light_buffer);
}


void updateCamera()
{
	const float vfov = 60.0f;
	const float aspect_ratio = static_cast<float>(width) /
		static_cast<float>(height);

	float3 camera_u, camera_v, camera_w;
	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		camera_u, camera_v, camera_w, true);

	const Matrix4x4 frame = Matrix4x4::fromBasis(
		normalize(camera_u),
		normalize(camera_v),
		normalize(-camera_w),
		camera_lookat);
	const Matrix4x4 frame_inv = frame.inverse();
	// Apply camera rotation twice to match old SDK behavior
	const Matrix4x4 trans = frame * camera_rotate * camera_rotate * frame_inv;

	camera_eye = make_float3(trans * make_float4(camera_eye, 1.0f));
	camera_lookat = make_float3(trans * make_float4(camera_lookat, 1.0f));
	camera_up = make_float3(trans * make_float4(camera_up, 0.0f));

	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		camera_u, camera_v, camera_w, true);

	camera_rotate = Matrix4x4::identity();

	context["eye"]->setFloat(camera_eye);
	context["U"]->setFloat(camera_u);
	context["V"]->setFloat(camera_v);
	context["W"]->setFloat(camera_w);
}



void glutInitialize(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(SAMPLE_NAME);
	glutHideWindow();
}


void glutRun()
{
	// Initialize GL state
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, width, height);

	glutShowWindow();
	glutReshapeWindow(width, height);

	// register glut callbacks
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutDisplay);
	glutReshapeFunc(glutResize);
	glutKeyboardFunc(glutKeyboardPress);
	glutMouseFunc(glutMousePress);
	glutMotionFunc(glutMouseMotion);

	registerExitHandler();

	glutMainLoop();
}

//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------
int part = 72*5;
void glutDisplay()
{
	updateCamera();
	context->launch(0, width, height);

	Buffer buffer = getOutputBuffer();
	sutil::displayBufferGL(getOutputBuffer());
	float m[16];
	if (rotation) {
		static int ii =0;
		if (ii < part) {

			char buf[32];
			sprintf(buf, "%03d", ii);
			char fol[8];
			sprintf(fol, "./%d/", part);
			const std::string outputImage =  fol + std::string(SAMPLE_NAME) + "_" + buf + ".ppm";
			std::cerr << "Saving current frame to '" << outputImage << "'\n";
			sutil::displayBufferPPM(outputImage.c_str(), getOutputBuffer());
			ii++;
		}
		t = 1;
		float angle = 2.0f/part * t * M_PIf;
		transform->getMatrix(false, m, NULL);

		/*float m[16] = { cosf(angle),0,sinf(angle),0,
						0,1,0,0,
						-sinf(angle),0,cosf(angle),0,
						0,0,0,1 };*/
		std::cout << meshmove << std::endl;

		float mm[16] = { cosf(angle),0,sinf(angle),0,
						0,1,0,0,
						-sinf(angle),0,cosf(angle),0,
						0,0,0,1 };
		Matrix4x4 tmat(m);
		Matrix4x4 rmat(mm);

		Matrix4x4 omat = rmat*tmat;
		transform->setMatrix(false, omat.getData(), NULL);
		//transform->setMatrix(false, rmat.getData(), NULL);
		// mark dirty so that the acceleration structure gets refit
		root_group->getAcceleration()->markDirty();

	}

	{
		static unsigned frame_count = 0;
		sutil::displayFps(frame_count++);
	}

	glutSwapBuffers();
}

static int calicount = 0;
void glutKeyboardPress(unsigned char k, int x, int y)
{
	// float m[16];
	switch (k)
	{
	case('q'):
	case(27): // ESC
	{
		destroyContext();
		exit(0);
	}
	case('s'):
	{
		//const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
		char buf[80];
		sprintf(buf, "c%02d.ppm", calicount++);
		std::string outputImage = std::string(buf);
		std::cerr << "Saving current frame to '" << outputImage << "'\n";
		sutil::displayBufferPPM(outputImage.c_str(), getOutputBuffer());
		break;
	}
	case('p'):
	{
		rotation = !rotation;
	}
	}
}


void glutMousePress(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_button = button;
		mouse_prev_pos = make_int2(x, y);
	}
	else
	{
		// nothing
	}
}


void glutMouseMotion(int x, int y)
{
	if (mouse_button == GLUT_RIGHT_BUTTON)
	{
		const float dx = static_cast<float>(x - mouse_prev_pos.x) /
			static_cast<float>(width);
		const float dy = static_cast<float>(y - mouse_prev_pos.y) /
			static_cast<float>(height);
		const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
		const float scale = fminf(dmax, 0.9f);
		camera_eye = camera_eye + (camera_lookat - camera_eye) * scale;
	}
	else if (mouse_button == GLUT_LEFT_BUTTON)
	{
		const float2 from = { static_cast<float>(mouse_prev_pos.x),
							  static_cast<float>(mouse_prev_pos.y) };
		const float2 to = { static_cast<float>(x),
							  static_cast<float>(y) };

		const float2 a = { from.x / width, from.y / height };
		const float2 b = { to.x / width, to.y / height };

		camera_rotate = arcball.rotate(b, a);
	}

	mouse_prev_pos = make_int2(x, y);
}


void glutResize(int w, int h)
{
	if (w == (int)width && h == (int)height) return;

	width = w;
	height = h;
	sutil::ensureMinimumSize(width, height);

	sutil::resizeBuffer(getOutputBuffer(), width, height);

	glViewport(0, 0, width, height);

	glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit(const std::string& argv0)
{
	std::cerr << "\nUsage: " << argv0 << " [options]\n";
	std::cerr <<
		"App Options:\n"
		"  -h | --help         Print this usage message and exit.\n"
		"  -f | --file         Save single frame to file and exit.\n"
		"  -n | --nopbo        Disable GL interop for display buffer.\n"
		"  -T | --tutorial-number <num>              Specify tutorial number\n"
		"  -t | --texture-path <path>                Specify path to texture directory\n"
		"App Keystrokes:\n"
		"  q  Quit\n"
		"  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
		<< std::endl;

	exit(1);
}

int main(int argc, char** argv)
{
	std::string out_file;
	for (int i = 1; i < argc; ++i)
	{
		const std::string arg(argv[i]);

		if (arg == "-h" || arg == "--help")
		{
			printUsageAndExit(argv[0]);
		}
	}

	if (texture_path.empty()) {
		texture_path = std::string(sutil::samplesDir()) + "/data";
	}

	try
	{
		glutInitialize(&argc, argv);

#ifndef __APPLE__
		glewInit();
#endif

		// load the ptx source associated with tutorial number
		std::stringstream ss;
		ss << "tutorial" << tutorial_number << ".cu";
		std::string tutorial_ptx_path = ss.str();
		tutorial_ptx = sutil::getPtxString(SAMPLE_NAME, tutorial_ptx_path.c_str());

		createContext();
		createGeometry();
		setupCamera();
		setupLights();

		context->validate();

		if (out_file.empty())
		{
			glutRun();
		}
		else
		{
			updateCamera();
			context->launch(0, width, height);
			sutil::displayBufferPPM(out_file.c_str(), getOutputBuffer());
			destroyContext();
		}
		return 0;
	}
	SUTIL_CATCH(context->get())
}

