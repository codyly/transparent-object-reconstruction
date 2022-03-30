
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

#define REFLECT_DEPTH	1
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

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(uint2,        launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned int, sample, , );

rtBuffer<unsigned int, 2>       rnd_seeds;

rtTextureSampler<float4, 2>		envmap;

struct RayInfo
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
	float   importance;
};

rtDeclareVariable(RayInfo, rayinfo, rtPayload, );

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

static __device__ __inline__ float3 exp( const float3& x )
{
  return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

// -----------------------------------------------------------------------------

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

float3 __device__ __inline__ sampleEnv(float3 dir)
{
	float u = atan2f(dir.x, dir.z) * M_1_PIf;
	float v = 1.0 - dir.y;
	return (v < 0) ? make_float3(.1, .1, .1) : make_float3( tex2D(envmap, u, v) );
}

RT_PROGRAM void trace_surface ()
{
	// parameters

	float importance_cutoff = 1e-2f;
	float3 cutoff_color = make_float3(0.34f, 0.55f, 0.85f);
	float fresnel_exponent = 3.0f;
	float fresnel_minimum = 0.1f;
	float fresnel_maximum = 1.0f;
	float refraction_index = 1.4f;
	float3 refraction_color = make_float3(1.0f, 1.0f, 1.0f);
	float3 reflection_color = make_float3(1.0f, 1.0f, 1.0f);
	int refraction_maxdepth = 100;
	int reflection_maxdepth = 100;
	float3 extinction_constant = make_float3(__logf(0.80f), __logf(0.89f), __logf(0.75f));
	float3 shadow_attenuation = make_float3(0.4f, 0.7f, 0.4f);
	
	// geometry vectors
	const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal  
	const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);
	const float3 bhp = rtTransformPoint(RT_OBJECT_TO_WORLD, back_hit_point);
	const float3 raydir = ray.direction;                                            // incident direction
	float3 lightdir, spos, refldir, refrdir, reflclr, refrclr, shadowclr;
	float3 jit = jitter_sample();
	float ndotl, refldist, refrdist;

	if (isnan(fhp.x) || isnan(raydir.x) ) return;

	float d = length(fhp - ray.origin);	
	lightdir = normalize(normalize(light_pos - fhp) + jit * mat.light_width );
	ndotl = dot(n, lightdir);

	// shading			
	float3 diffuse		= mat.diff_color * sampleEnv ( lightdir ) * max(0.0f, ndotl );
	float3 spec			= mat.spec_color * pow( max(0.0f, dot( n, normalize(-raydir + normalize(light_pos-fhp)))), (float) mat.spec_power );
	
	reflclr = make_float3(0, 0, 0);
	refrclr = make_float3(0, 0, 0);
	shadowclr = make_float3(1, 1, 1);

	if (rayinfo.depth < REFLECT_DEPTH && mat.refl_width > 0) {			
		// reflection sample					
		refldir = normalize(normalize(2 * dot(n, -raydir) * n + raydir) + jit * mat.refl_width);
		reflclr = TraceRay(fhp + refldir*mat.refl_bias, refldir, rayinfo.depth + 1, ANY_RAY, refldist) * mat.refl_color;
	}
	
	if (rayinfo.depth < REFRACT_DEPTH && mat.refr_width > 0) {
		// refraction sample
		optix::refract(refrdir, raydir, n, mat.refr_ior);
		refrdir = normalize(refrdir);// + jit * mat.refr_width);
		if (!isnan(refrdir.x)) {
			refrclr = TraceRay(fhp + refrdir*mat.refr_bias, refrdir, rayinfo.depth + 1, REFRACT_RAY, refrdist);
			// refrclr = lerp3(mat.refr_color, refrclr, refrdist / mat.refr_offset);			
			// refrclr = lerp3(refrclr*mat.refr_amount, mat.refr_color,  min(1.0f, refrdist / mat.refr_offset) );
		}
	}
	// if (rayinfo.depth < SHADOW_DEPTH) {
	// 	// shadow sample		
	// 	for (int i = 0; i < 2; i++) {
	// 		lightdir = normalize(normalize(light_pos - fhp) + jitter_sample() * mat.light_width);
	// 		shadowclr *= TraceRay(fhp + lightdir*mat.shadow_bias, lightdir, rayinfo.depth + 1, SHADOW_RAY, refldist);
	// 	}
	// }
	if (mat.env_color.x == 1) {
		float chk = ((int(floor(fhp.x / mat.env_color.y) + floor(fhp.z / mat.env_color.y)) & 1) == 0) ? 1.0 : mat.env_color.z;
		diffuse *= chk;

	}

	rayinfo.result = (diffuse*make_float3(.85f, .85f, .85f) + spec + mat.amb_color)*shadowclr.x + (reflclr + refrclr)*(shadowclr.x*0.3+0.7);
	// rayinfo.result = reflclr + refrclr;
	// printf("%f,%f,%f\n",raydir.x, raydir.y, raydir.z);
	rayinfo.length = d;
	rayinfo.alpha = .9f;
	// rayinfo.alpha = 0; // deep_color.w;
	// rayinfo.matid = hit_mat;
}


// -----------------------------------------------------------------------------

//
// Attenuates shadow rays for shadowing transparent objects
//

RT_PROGRAM void trace_shadow ()
{
	float3 shadow_attenuation = make_float3(0.4f, 0.4f, 0.4f);
	float3 world_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float nDi = fabs(dot(world_normal, ray.direction));

	rayinfo.alpha *= (1-fresnel_schlick(nDi, 5, 1-shadow_attenuation, make_float3(1))).x;

	rtIgnoreIntersection();	
}


static __device__ __inline__ float3 schlick( float nDi, const float3& rgb )
{
	float r = fresnel_schlick(nDi, 5, rgb.x, 1);
	float g = fresnel_schlick(nDi, 5, rgb.y, 1);
	float b = fresnel_schlick(nDi, 5, rgb.z, 1);
	return make_float3(r, g, b);
}


//
// (NEW)
// Attenuates shadow rays for shadowing transparent objects
//

RT_PROGRAM void glass_any_hit_shadow()
{
  float3 shadow_attenuation = make_float3(0.4f, 0.7f, 0.4f);
  float3 world_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float nDi = fabs(dot(world_normal, ray.direction));

  rayinfo.alpha *= (1-fresnel_schlick(nDi, 5, 1-shadow_attenuation, make_float3(1))).x;

  rtIgnoreIntersection();			
}


// //
// // Dielectric surface shader
// //
// rtDeclareVariable(float3,       cutoff_color, , );
// rtDeclareVariable(float,        fresnel_exponent, , );
// rtDeclareVariable(float,        fresnel_minimum, , );
// rtDeclareVariable(float,        fresnel_maximum, , );
// rtDeclareVariable(float,        refraction_index, , );
// rtDeclareVariable(int,          refraction_maxdepth, , );
// rtDeclareVariable(int,          reflection_maxdepth, , );
// rtDeclareVariable(float3,       refraction_color, , );
// rtDeclareVariable(float3,       reflection_color, , );
// rtDeclareVariable(float3,       extinction_constant, , );
RT_PROGRAM void glass_closest_hit_radiance()
{
  // intersection vectors
  float importance_cutoff = 1e-2f;
  float3 cutoff_color = make_float3(0.34f, 0.55f, 0.85f);
  float fresnel_exponent = 3.0f;
  float fresnel_minimum = 0.1f;
  float fresnel_maximum = 1.0f;
  float refraction_index = 1.4f;
  float3 refraction_color = make_float3(1.0f, 1.0f, 1.0f);
  float3 reflection_color = make_float3(1.0f, 1.0f, 1.0f);
  int refraction_maxdepth = 100;
  int reflection_maxdepth = 100;
  float3 extinction_constant = make_float3(__logf(0.80f), __logf(0.89f), __logf(0.75f));
  float3 shadow_attenuation = make_float3(0.4f, 0.7f, 0.4f);

	// const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);
  const float3 h = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);           // hitpoint
  const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
  const float3 i = ray.direction;                                            // incident direction

  float reflection = 1.0f;
  float3 result = make_float3(0.0f);

  float3 beer_attenuation;
  if(dot(n, ray.direction) > 0){
    // Beer's law attenuation
    beer_attenuation = exp(extinction_constant * t_hit);
  } else {
    beer_attenuation = make_float3(1);
  }

  // refraction
  if (rayinfo.depth < min(refraction_maxdepth, max_depth))
  {
    float3 t;                                                            // transmission direction
    if ( refract(t, i, n, refraction_index) )
    {

      // check for external or internal reflection
      float cos_theta = dot(i, n);
      if (cos_theta < 0.0f)
        cos_theta = -cos_theta;
      else
        cos_theta = dot(t, n);

      reflection = fresnel_schlick(cos_theta, fresnel_exponent, fresnel_minimum, fresnel_maximum);

      float importance = rayinfo.importance * (1.0f-reflection) * optix::luminance( refraction_color * beer_attenuation );
      if ( importance > importance_cutoff ) {
        optix::Ray ray( h, t, 0, 0.0f, RT_DEFAULT_MAX  );
        RayInfo refr_prd;
        refr_prd.depth = rayinfo.depth+1;
		refr_prd.importance = importance;
		refr_prd.alpha = 1.0f;
		refr_prd.length = 0;
		refr_prd.rtype = 0;


        rtTrace( top_object, ray, refr_prd );
        result += (1.0f - reflection) * refraction_color * refr_prd.result;
      } else {
        result += (1.0f - reflection) * refraction_color * cutoff_color;
      }
    }
    // else TIR
  }

  // reflection
  if (rayinfo.depth < min(reflection_maxdepth, max_depth))
  {
    float3 r = reflect(i, n);

    float importance = rayinfo.importance * reflection * optix::luminance( reflection_color * beer_attenuation );
    if ( importance > importance_cutoff ) {
      optix::Ray ray( h, r, 0, 0.0f, RT_DEFAULT_MAX  );
      RayInfo refl_prd;
      refl_prd.depth = rayinfo.depth+1;
	  refl_prd.importance = importance;
	  refl_prd.alpha = 1.0f;
	  refl_prd.length = 0;
	  refl_prd.rtype = 0;

      rtTrace( top_object, ray, refl_prd );
      result += reflection * reflection_color * refl_prd.result;
    } else {
      result += reflection * reflection_color * cutoff_color;
    }
  }

  result = result * beer_attenuation;

  rayinfo.result = result;
}