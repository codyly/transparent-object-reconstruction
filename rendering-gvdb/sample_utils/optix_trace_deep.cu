//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this 
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this 
//    list of conditions and the following disclaimer in the documentation and/or 
//    other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may 
//    be used to endorse or promote products derived from this software without specific 
//   prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//----------------------------------------------------------------------------------

#include "optix_extra_math.cuh"

#include "optix_gvdb.cuh"



#define REFLECT_DEPTH	3
#define REFRACT_DEPTH	2
#define SHADOW_DEPTH	5

#define ANY_RAY			0
#define	SHADOW_RAY		1
#define VOLUME_RAY		2
#define MESH_RAY		3
#define REFRACT_RAY		4

struct Material {
	char		name[64];
	int			id;
	float		light_width;		// light scatter
	
	float3		amb_color;
	float3		env_color;			// 0.5,0.5,0.5
	float3		diff_color;			// .6,.7,.7
	float3		spec_color;			// 3,3,3
	float		spec_power;			// 400		

	float		shadow_width;		// shadow scatter
	float		shadow_bias;

	float		refl_width;			// reflect scatter
	float3		refl_color;			// 1,1,1		
	float		refl_bias;

	float		refr_width;			// refract scatter
	float		refr_ior;			// 1.2
	float3		refr_color;			// .35, .4, .4
	float		refr_amount;		// 10
	float		refr_offset;		// 15
	float		refr_bias;
};

rtDeclareVariable(float3,       light_pos, , );
rtDeclareVariable(Material,		mat, , );

rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(unsigned int, shadow_enable, , );
rtDeclareVariable(unsigned int, mirror_enable, , );
rtDeclareVariable(unsigned int, cone_enable, , );
rtDeclareVariable(int,          max_depth, , );

rtDeclareVariable(float3,		shading_normal,		attribute shading_normal, ); 
rtDeclareVariable(float3,		front_hit_point,	attribute front_hit_point, );
rtDeclareVariable(float3,		back_hit_point,		attribute back_hit_point, );
rtDeclareVariable(float4,		deep_color,			attribute deep_color, );
rtDeclareVariable(int,			obj_type,			attribute obj_type, );
rtDeclareVariable(float3,       ref_dir,            attribute ref_dir, );

rtDeclareVariable(optix::Ray,   ray,          rtCurrentRay, );
rtDeclareVariable(float,        t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,        launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned int, sample, , );

rtBuffer<unsigned int, 2>       rnd_seeds;

struct RayInfo
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
	float   importance;
};

rtDeclareVariable(RayInfo,		rayinfo, rtPayload, );

// -----------------------------------------------------------------------------

static __device__ __inline__ float3 TraceRay (float3 origin, float3 direction, int depth, int rtype, float& length )
{
  optix::Ray ray = optix::make_Ray( origin, direction, 0, 0.0f, RT_DEFAULT_MAX );
  RayInfo rayi;
  rayi.length = 0.f;
  rayi.depth = depth;
  rayi.rtype = rtype;
  rayi.alpha = 1.f;
  rayi.importance = 1.f;
  rtTrace( top_object, ray, rayi );
  length = rayi.length;
  return (rtype == SHADOW_RAY) ? make_float3(rayi.alpha, rayi.alpha, rayi.alpha) : rayi.result;
}

float3 __device__ __inline__ jitter_sample ()
{	 
	uint2 index = make_uint2(launch_index.x & 0x7F, launch_index.y & 0x7F);
	unsigned int seed = rnd_seeds[index];  	
	float uu = rnd(seed) - 0.5f;
	float vv = rnd(seed) - 0.5f;
	float ww = rnd(seed) - 0.5f;
	rnd_seeds[index] = seed;
	return make_float3(uu, vv, ww);
}

int __device__ __inline__ gen_seed ()
{
	uint2 index = make_uint2(launch_index.x % 1280, launch_index.y % 1280);
	unsigned int seed = rnd_seeds[index];
	float a = rnd(seed); 
	rnd_seeds[index] = seed;
	return seed;
}


void __device__ __inline__ inverseT (float* data)
{
	double inv[16], det;
	// mult: 16 *  13 + 4 	= 212
	// add:   16 * 5 + 3 	=   83
	int i;
	inv[0] =   data[5]*data[10]*data[15] - data[5]*data[11]*data[14] - data[9]*data[6]*data[15] + data[9]*data[7]*data[14] + data[13]*data[6]*data[11] - data[13]*data[7]*data[10];
	inv[4] =  -data[4]*data[10]*data[15] + data[4]*data[11]*data[14] + data[8]*data[6]*data[15]- data[8]*data[7]*data[14] - data[12]*data[6]*data[11] + data[12]*data[7]*data[10];
	inv[8] =   data[4]*data[9]*data[15] - data[4]*data[11]*data[13] - data[8]*data[5]*data[15]+ data[8]*data[7]*data[13] + data[12]*data[5]*data[11] - data[12]*data[7]*data[9];
	inv[12] = -data[4]*data[9]*data[14] + data[4]*data[10]*data[13] + data[8]*data[5]*data[14]- data[8]*data[6]*data[13] - data[12]*data[5]*data[10] + data[12]*data[6]*data[9];
	inv[1] =  -data[1]*data[10]*data[15] + data[1]*data[11]*data[14] + data[9]*data[2]*data[15]- data[9]*data[3]*data[14] - data[13]*data[2]*data[11] + data[13]*data[3]*data[10];
	inv[5] =   data[0]*data[10]*data[15] - data[0]*data[11]*data[14] - data[8]*data[2]*data[15]+ data[8]*data[3]*data[14] + data[12]*data[2]*data[11] - data[12]*data[3]*data[10];
	inv[9] =  -data[0]*data[9]*data[15] + data[0]*data[11]*data[13] + data[8]*data[1]*data[15]- data[8]*data[3]*data[13] - data[12]*data[1]*data[11] + data[12]*data[3]*data[9];
	inv[13] =  data[0]*data[9]*data[14] - data[0]*data[10]*data[13] - data[8]*data[1]*data[14]+ data[8]*data[2]*data[13] + data[12]*data[1]*data[10] - data[12]*data[2]*data[9];
	inv[2] =   data[1]*data[6]*data[15] - data[1]*data[7]*data[14] - data[5]*data[2]*data[15]+ data[5]*data[3]*data[14] + data[13]*data[2]*data[7] - data[13]*data[3]*data[6];
	inv[6] =  -data[0]*data[6]*data[15] + data[0]*data[7]*data[14] + data[4]*data[2]*data[15]- data[4]*data[3]*data[14] - data[12]*data[2]*data[7] + data[12]*data[3]*data[6];
	inv[10] =  data[0]*data[5]*data[15] - data[0]*data[7]*data[13] - data[4]*data[1]*data[15]+ data[4]*data[3]*data[13] + data[12]*data[1]*data[7] - data[12]*data[3]*data[5];
	inv[14] = -data[0]*data[5]*data[14] + data[0]*data[6]*data[13] + data[4]*data[1]*data[14]- data[4]*data[2]*data[13] - data[12]*data[1]*data[6] + data[12]*data[2]*data[5];
	inv[3] =  -data[1]*data[6]*data[11] + data[1]*data[7]*data[10] + data[5]*data[2]*data[11]- data[5]*data[3]*data[10] - data[9]*data[2]*data[7] + data[9]*data[3]*data[6];
	inv[7] =   data[0]*data[6]*data[11] - data[0]*data[7]*data[10] - data[4]*data[2]*data[11]+ data[4]*data[3]*data[10] + data[8]*data[2]*data[7] - data[8]*data[3]*data[6];
	inv[11] = -data[0]*data[5]*data[11] + data[0]*data[7]*data[9] + data[4]*data[1]*data[11]- data[4]*data[3]*data[9] - data[8]*data[1]*data[7] + data[8]*data[3]*data[5];
	inv[15] =  data[0]*data[5]*data[10] - data[0]*data[6]*data[9] - data[4]*data[1]*data[10]+ data[4]*data[2]*data[9] + data[8]*data[1]*data[6] - data[8]*data[2]*data[5];
	
	det = data[0]*inv[0] + data[1]*inv[4] + data[2]*inv[8] + data[3]*inv[12];
	if (det == 0)    return;
	det = 1.0f / det;

	for (i = 0; i < 16; i++)  
		data[i] = (float) (inv[i] * det);
	
	return;
}


float3 __device__ __inline__ getCubicNorm(float3 coord, float3 center)
{
	float3 rCoord = coord - center;
	float3 rabs = fabs3(rCoord);
	float3 mask = make_float3(0,0,0);
	mask.x = (rabs.x >= rabs.y) ? ((rabs.x >= rabs.z) ? 1 : 0) : 0;
	mask.y = (rabs.y >= rabs.x) ? ((rabs.y >= rabs.z) ? 1 : 0) : 0;
	mask.z = (rabs.z >= rabs.x) ? ((rabs.z >= rabs.y) ? 1 : 0) : 0;
	float3 n = numericmult(mask, fsign3(rCoord));
	return n;
}


RT_PROGRAM void trace_deep ()
{
	// Volumetric material

	// We arrive here after vol_deep has already traced into the gvdb volume.
	// - deep_color is the accumulated color along the volume ray
	// - front_hit_point is the start point of the volume
	// - back_hit_point is the ending point of the volume
	float3 lightPosition_base = make_float3(20, 20, -5000);
	float3 lightScreen_norm = make_float3(0,0,1);
	float3 lighta = make_float3(0, 25, 0);
	float3 lightb = make_float3(25, 0, 0);
	float alength = 25, blength = 25;
	int  precision = 4;


	float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal  
	const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);
	const float3 bhp = rtTransformPoint(RT_OBJECT_TO_WORLD, back_hit_point);
	const float3 raydir = ray.direction;                                            // incident direction
	float3 lightdir, spos, refldir, reflclr, shadowclr;
	float3 jit = jitter_sample();
	float ndotl, refldist;

	n = getCubicNorm(fhp, make_float3(0,0,0));


	if (isnan(fhp.x) || isnan(raydir.x) || isnan(n.x)) return;

	float d = length(fhp - ray.origin);	
	lightdir = normalize(normalize(light_pos - fhp) + jit * mat.light_width );
	ndotl = dot(n, lightdir);

	// Blending with polygonal objects 
	float plen;		
	float3 pos = ray.origin; 
	float3 bgclr = make_float3(0,0,0);// = TraceRay ( pos, ray.direction, rayinfo.depth, MESH_RAY, plen );	// poly ray
	// float3 bgclr = TraceRay ( pos, ref_dir, rayinfo.depth, MESH_RAY, plen );
	// float3 bgclr = make_float3(1.0f,1.0f,1.0f);
	// the background trace ray's direction shouldn't be straight.

	reflclr = make_float3(0, 0, 0);
	shadowclr = make_float3(1, 1, 1);

	if (!isnan(n.x) && rayinfo.depth < REFLECT_DEPTH && mat.refl_width > 0) {			
		// reflection sample					
		refldir = normalize(normalize(2 * dot(n, -raydir) * n + raydir) + jit * mat.refl_width);
		reflclr = TraceRay(fhp + refldir*mat.refl_bias, refldir, rayinfo.depth + 1, MESH_RAY, refldist) * mat.refl_color;
	}
	
	float vlen = length(front_hit_point - ray.origin);		// volume ray
	float a = deep_color.w;
	float3 mix_color;
	float3 rdir = ray.direction, rpos, rfilter, hit = make_float3(0,0,NOHIT), norm = make_float3(0,1,0);
	float4 clr = make_float4(0,0,0,1.0f);
	//if (plen < vlen) { a = 0; vlen = plen; }
	rayinfo.length = vlen;
	rayinfo.alpha = 0.0f;
	// rayinfo.result = bgclr;
	float3 pos_rc = mmult(pos, SCN_INVXFORM);
	float xform_o[16], x_form_n[16];
	for(int i=0; i<16; i++) x_form_n[i] = SCN_INVXFORM[i];
	inverseT(x_form_n);
	float3 dir_rc = mmult(normalize(ray.direction), SCN_INVXROT);
	float3 btdf_clr = make_float3(0, 0, 0);
	float3 brdf_clr = reflclr;
	float3 epsilon = make_float3(1e-2, 1e-2, 1e-2);
	float3 illum = make_float3(0, 0, 0);
	int i=1;
	unsigned int seed = gen_seed();

	// printf("pos: %.5f\n", pos.z );
	while (i <= 10){
		// dir_rc += epsilon;`
		hit = make_float3(0,0,NOHIT);
		norm = make_float3(0,1,0);
		clr = make_float4(0,0,0,1);
		rdir.x = float(seed);
		
		float pplen;
		if (rayinfo.rtype == MESH_RAY ) continue;
		
		rayCast ( &gvdbObj, gvdbChan, pos_rc, dir_rc, hit, norm, clr, rayDeepBrick, rdir, rpos, rfilter);

		if ( hit.x==0 && hit.y == 0){
			continue;
		}


		for(int i=0; i<16; i++) xform_o[i] = SCN_INVXFORM[i];
		inverseT(xform_o);
		rpos = mmult(rpos, xform_o);

		float3 rpos_w =  rtTransformPoint(RT_OBJECT_TO_WORLD, rpos);
		float3 rn = getCubicNorm(fhp, make_float3(0,0,0));
		
		if(i == 0){
			printf("rpos_w: x:%.3f, y:%.3f, z:%.3f\n", rpos_w.x, rpos_w.y, rpos_w.z);
		}

		float3 light_border = 0.5f * (lightPosition_base * 2.0f - lighta - lightb);
		illum = make_float3(0,0,0);
		for(int da=0; da<precision; da++)
		{
			for(int db=0; db < precision; db ++)
			{
				float3 tg_light = light_border + (float) da * lighta / precision + (float) db * lightb / precision;
				illum += (make_float3(1.0, 1.0, 1.0) / 4 / 3.1415926535 / dot(tg_light-rpos_w, tg_light-rpos_w)) ;
			}
		}
		illum *= 1e3 * 0.05f;

		float3 new_ray_dir = normalize(mmult(rdir, SCN_XROT));

		if(!isnan(new_ray_dir.x)){
			bgclr = TraceRay ( rpos, new_ray_dir, rayinfo.depth, MESH_RAY, pplen );
			mix_color = ( illum + bgclr )*rfilter;
			// Result is blending of background/poly color and the volume color (deep_color)		
			btdf_clr = lerp3 ( btdf_clr, mix_color, 1.0f/(float)(i));
			i++;
		}
	}
	rayinfo.result = lerp3(brdf_clr, btdf_clr, .9f) * 0.8f;
	
	// prd_radiance.result = fhp/200.0;			-- debugging
}

// -----------------------------------------------------------------------------

//
// Attenuates shadow rays for shadowing transparent objects
//
RT_PROGRAM void trace_shadow ()
{
	// rtype is SHADOW_RAY
	rayinfo.alpha = deep_color.w;
}
