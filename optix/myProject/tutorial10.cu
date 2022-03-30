/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "tutorial.h"
#include <optixu/optixu_aabb.h>

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );




//
// Pinhole camera implementation
//
rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtBuffer<uchar4, 2>              output_buffer;

RT_PROGRAM void pinhole_camera()
{
	size_t2 screen = output_buffer.size();

	float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x * U + d.y * V + W);

	optix::Ray ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon);

	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.depth = 0;

	rtTrace(top_object, ray, prd);

	output_buffer[launch_index] = make_color(prd.result);
}


//
// Environment map background
//
rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
	float theta = atan2f(ray.direction.x, ray.direction.z);
	float phi = M_PIf * 0.5f - acosf(ray.direction.y);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sin(phi));
	prd_radiance.result = make_float3(tex2D(envmap, u, v));
}


//
// Terminates and fully attenuates ray after any hit
//
RT_PROGRAM void any_hit_shadow()
{
	// this material is opaque, so it fully attenuates all shadow rays
	prd_shadow.attenuation = make_float3(0);

	rtTerminateRay();
}


//
// Procedural rusted metal surface shader
//

/*
 * Translated to CUDA C from Larry Gritz's LGRustyMetal.sl shader found at:
 * http://renderman.org/RMR/Shaders/LGShaders/LGRustyMetal.sl
 *
 * Used with permission from tal AT renderman DOT org.
 */

rtDeclareVariable(float3, ambient_light_color, , );
rtBuffer<BasicLight>        lights;
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float, importance_cutoff, , );
rtDeclareVariable(int, max_depth, , );
rtDeclareVariable(float3, reflectivity_n, , );

rtDeclareVariable(float, metalKa, , ) = 1;
rtDeclareVariable(float, metalKs, , ) = 1;
rtDeclareVariable(float, metalroughness, , ) = .1;
rtDeclareVariable(float, rustKa, , ) = 1;
rtDeclareVariable(float, rustKd, , ) = 1;
rtDeclareVariable(float3, rustcolor, , ) = { .437, .084, 0 };
rtDeclareVariable(float3, metalcolor, , ) = { .7, .7, .7 };
rtDeclareVariable(float, txtscale, , ) = .02;
rtDeclareVariable(float, rusty, , ) = 0.2;
rtDeclareVariable(float, rustbump, , ) = 0.85;
#define MAXOCTAVES 6

rtTextureSampler<float, 3> noise_texture;
static __device__ __inline__ float snoise(float3 p)
{
	return tex3D(noise_texture, p.x, p.y, p.z) * 2 - 1;
}


RT_PROGRAM void box_closest_hit_radiance()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 hit_point = ray.origin + t_hit * ray.direction;

	/* Sum several octaves of abs(snoise), i.e. turbulence.  Limit the
	 * number of octaves by the estimated change in PP between adjacent
	 * shading samples.
	 */
	float3 PP = txtscale * hit_point;
	float a = 1;
	float sum = 0;
	for (int i = 0; i < MAXOCTAVES; i++) {
		sum += a * fabs(snoise(PP));
		PP *= 2.0f;
		a *= 0.5f;
	}

	/* Scale the rust appropriately, modulate it by another noise
	 * computation, then sharpen it by squaring its value.
	 */
	float rustiness = step(1 - rusty, clamp(sum, 0.0f, 1.0f));
	rustiness *= clamp(abs(snoise(PP)), 0.0f, .08f) / 0.08f;
	rustiness *= rustiness;
	rustiness = 0;
	/* If we have any rust, calculate the color of the rust, taking into
	 * account the perturbed normal and shading like matte.
	 */
	float3 Nrust = ffnormal;
	if (rustiness > 0) {
		/* If it's rusty, also add a high frequency bumpiness to the normal */
		Nrust = normalize(ffnormal + rustbump * snoise(PP));
		Nrust = faceforward(Nrust, -ray.direction, world_geo_normal);
	}

	float3 color = mix(metalcolor * metalKa, rustcolor * rustKa, rustiness) * ambient_light_color;
	for (int i = 0; i < lights.size(); ++i) {
		BasicLight light = lights[i];
		float3 L = normalize(light.pos - hit_point);
		float nmDl = dot(ffnormal, L);
		float nrDl = dot(Nrust, L);

		if (nmDl > 0.0f || nrDl > 0.0f) {
			// cast shadow ray
			PerRayData_shadow shadow_prd;
			shadow_prd.attenuation = make_float3(1.0f);
			float Ldist = length(light.pos - hit_point);
			optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist);
			rtTrace(top_shadower, shadow_ray, shadow_prd);
			float3 light_attenuation = shadow_prd.attenuation;

			if (fmaxf(light_attenuation) > 0.0f) {
				float3 Lc = light.color * light_attenuation;
				nrDl = max(nrDl * rustiness, 0.0f);
				color += rustKd * rustcolor * nrDl * Lc;

				float r = nmDl * (1.0f - rustiness);
				if (nmDl > 0.0f) {
					float3 H = normalize(L - ray.direction);
					float nmDh = dot(ffnormal, H);
					if (nmDh > 0)
						color += r * metalKs * Lc * pow(nmDh, 1.f / metalroughness);
				}
			}

		}
	}

	float3 r = schlick(-dot(ffnormal, ray.direction), reflectivity_n * (1 - rustiness));
	float importance = prd_radiance.importance * optix::luminance(r);

	// reflection ray
	if (importance > importance_cutoff && prd_radiance.depth < max_depth) {
		PerRayData_radiance refl_prd;
		refl_prd.importance = importance;
		refl_prd.depth = prd_radiance.depth + 1;
		float3 R = reflect(ray.direction, ffnormal);
		optix::Ray refl_ray(hit_point, R, RADIANCE_RAY_TYPE, scene_epsilon);
		rtTrace(top_object, refl_ray, refl_prd);
		color += r * refl_prd.result;
	}

	prd_radiance.result = color;
	prd_radiance.result = make_float3(1.0f, 1.0f, 1.0f);
}


//
// Phong surface shading with shadows and schlick-approximated fresnel reflections.
// Uses procedural texture to determine diffuse response.
//
rtDeclareVariable(float, phong_exp, , );
rtDeclareVariable(float3, tile_v0, , );
rtDeclareVariable(float3, tile_v1, , );
rtDeclareVariable(float3, crack_color, , );
rtDeclareVariable(float, crack_width, , );
rtDeclareVariable(float3, Ka, , );
rtDeclareVariable(float3, Ks, , );
rtDeclareVariable(float3, Kd, , );

RT_PROGRAM void floor_closest_hit_radiance()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 color = Ka * ambient_light_color;

	float3 hit_point = ray.origin + t_hit * ray.direction;

	float v0 = dot(tile_v0, hit_point);
	float v1 = dot(tile_v1, hit_point);
	v0 = v0 - floor(v0);
	v1 = v1 - floor(v1);

	float3 local_Kd;
	if (v0 > crack_width && v1 > crack_width) {
		local_Kd = Kd;
	}
	else {
		local_Kd = crack_color;
	}

	for (int i = 0; i < lights.size(); ++i) {
		BasicLight light = lights[i];
		float3 L = normalize(light.pos - hit_point);
		float nDl = dot(ffnormal, L);

		if (nDl > 0.0f) {
			// cast shadow ray
			PerRayData_shadow shadow_prd;
			shadow_prd.attenuation = make_float3(1.0f);
			float Ldist = length(light.pos - hit_point);
			optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist);
			rtTrace(top_shadower, shadow_ray, shadow_prd);
			float3 light_attenuation = shadow_prd.attenuation;

			if (fmaxf(light_attenuation) > 0.0f) {
				float3 Lc = light.color * light_attenuation;
				color += local_Kd * nDl * Lc;

				float3 H = normalize(L - ray.direction);
				float nDh = dot(ffnormal, H);
				if (nDh > 0)
					color += Ks * Lc * pow(nDh, phong_exp);
			}

		}
	}

	float3 r = schlick(-dot(ffnormal, ray.direction), reflectivity_n);
	float importance = prd_radiance.importance * optix::luminance(r);

	// reflection ray
	if (importance > importance_cutoff && prd_radiance.depth < max_depth) {
		PerRayData_radiance refl_prd;
		refl_prd.importance = importance;
		refl_prd.depth = prd_radiance.depth + 1;
		float3 R = reflect(ray.direction, ffnormal);
		optix::Ray refl_ray(hit_point, R, RADIANCE_RAY_TYPE, scene_epsilon);
		rtTrace(top_object, refl_ray, refl_prd);
		color += r * refl_prd.result;
	}

	prd_radiance.result = color;
}


//
// (NEW)
// Attenuates shadow rays for shadowing transparent objects
//

rtDeclareVariable(float3, shadow_attenuation, , );

RT_PROGRAM void glass_any_hit_shadow()
{
	float3 world_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float nDi = fabs(dot(world_normal, ray.direction));

	prd_shadow.attenuation *= 1 - fresnel_schlick(nDi, 5, 1 - shadow_attenuation, make_float3(1));

	rtIgnoreIntersection();
}


//
// Dielectric surface shader
//
rtDeclareVariable(float3, cutoff_color, , );
rtDeclareVariable(float, fresnel_exponent, , );
rtDeclareVariable(float, fresnel_minimum, , );
rtDeclareVariable(float, fresnel_maximum, , );
rtDeclareVariable(float, refraction_index, , );
rtDeclareVariable(int, refraction_maxdepth, , );
rtDeclareVariable(int, reflection_maxdepth, , );
rtDeclareVariable(float3, refraction_color, , );
rtDeclareVariable(float3, reflection_color, , );
rtDeclareVariable(float3, extinction_constant, , );
RT_PROGRAM void glass_closest_hit_radiance()
{
	// intersection vectors
	const float3 h = ray.origin + t_hit * ray.direction;            // hitpoint
	const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
	const float3 i = ray.direction;                                            // incident direction

	float reflection = 1.0f;
	float3 result = make_float3(0.0f);

	float3 beer_attenuation;
	if (dot(n, ray.direction) > 0) {
		// Beer's law attenuation
		beer_attenuation = exp(extinction_constant * t_hit);
	}
	else {
		beer_attenuation = make_float3(1);
	}

	// refraction
	if (prd_radiance.depth < min(refraction_maxdepth, max_depth))
	{
		float3 t;                                                            // transmission direction
		if (refract(t, i, n, refraction_index))
		{

			// check for external or internal reflection
			float cos_theta = dot(i, n);
			if (cos_theta < 0.0f)
				cos_theta = -cos_theta;
			else
				cos_theta = dot(t, n);

			reflection = fresnel_schlick(cos_theta, fresnel_exponent, fresnel_minimum, fresnel_maximum);

			float importance = prd_radiance.importance * (1.0f - reflection) * optix::luminance(refraction_color * beer_attenuation);
			if (importance > importance_cutoff) {
				optix::Ray ray(h, t, RADIANCE_RAY_TYPE, scene_epsilon);
				PerRayData_radiance refr_prd;
				refr_prd.depth = prd_radiance.depth + 1;
				refr_prd.importance = importance;

				rtTrace(top_object, ray, refr_prd);
				result += (1.0f - reflection) * refraction_color * refr_prd.result;
			}
			else {
				result += (1.0f - reflection) * refraction_color * cutoff_color;
			}
		}
		// else TIR
	}

	// reflection
	if (prd_radiance.depth < min(reflection_maxdepth, max_depth))
	{
		float3 r = reflect(i, n);

		float importance = prd_radiance.importance * reflection * optix::luminance(reflection_color * beer_attenuation);
		if (importance > importance_cutoff) {
			optix::Ray ray(h, r, RADIANCE_RAY_TYPE, scene_epsilon);
			PerRayData_radiance refl_prd;
			refl_prd.depth = prd_radiance.depth + 1;
			refl_prd.importance = importance;

			rtTrace(top_object, ray, refl_prd);
			result += reflection * reflection_color * refl_prd.result;
		}
		else {
			result += reflection * reflection_color * cutoff_color;
		}
	}

	result = result * beer_attenuation;

	prd_radiance.result = result;
}


//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
	output_buffer[launch_index] = make_color(bad_color);
}


RT_PROGRAM void closest_hit_radiance3()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 color = Ka * ambient_light_color;

	float3 hit_point = ray.origin + t_hit * ray.direction;

	for (int i = 0; i < lights.size(); ++i) {
		BasicLight light = lights[i];
		float3 L = normalize(light.pos - hit_point);
		float nDl = dot(ffnormal, L);

		if (nDl > 0.0f) {
			// cast shadow ray
			PerRayData_shadow shadow_prd;
			shadow_prd.attenuation = make_float3(1.0f);
			float Ldist = length(light.pos - hit_point);
			optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist);
			rtTrace(top_shadower, shadow_ray, shadow_prd);
			float3 light_attenuation = shadow_prd.attenuation;

			if (fmaxf(light_attenuation) > 0.0f) {
				float3 Lc = light.color * light_attenuation;
				color += Kd * nDl * Lc;

				float3 H = normalize(L - ray.direction);
				float nDh = dot(ffnormal, H);
				if (nDh > 0)
					color += Ks * Lc * pow(nDh, phong_exp);
			}

		}
	}
	prd_radiance.result = color;
}


RT_PROGRAM void screen_hit()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 color = Ka * ambient_light_color;

	float3 hit_point = ray.origin + t_hit * ray.direction;

	for (int i = 0; i < lights.size(); ++i) {
		BasicLight light = lights[i];
		float3 L = normalize(light.pos - hit_point);
		float nDl = dot(ffnormal, L);

		if (nDl > 0.0f) {
			// cast shadow ray
			PerRayData_shadow shadow_prd;
			shadow_prd.attenuation = make_float3(1.0f);
			float Ldist = length(light.pos - hit_point);
			optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist);
			rtTrace(top_shadower, shadow_ray, shadow_prd);
			float3 light_attenuation = shadow_prd.attenuation;

			if (fmaxf(light_attenuation) > 0.0f) {
				float3 Lc = light.color * light_attenuation;
				color += Kd * nDl * Lc;

				float3 H = normalize(L - ray.direction);
				float nDh = dot(ffnormal, H);
				if (nDh > 0)
					color += Ks * Lc * pow(nDh, phong_exp);
			}

		}
	}
	prd_radiance.result = Ka;
}


//
// (UPDATED)
// Phong surface shading with shadows 
//

rtTextureSampler<float4, 2> tagsmap;

RT_PROGRAM void closest_hit_tags()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 color = Ka * ambient_light_color;

	float3 hit_point = ray.origin + t_hit * ray.direction;

	float u = (-hit_point.z+80) / 160.0f;
	float v = (hit_point.y-80) / 160.0f;
	color = make_float3(tex2D(tagsmap, u, v));

	for (int i = 0; i < lights.size(); ++i) {
		BasicLight light = lights[i];
		float3 L = normalize(light.pos - hit_point);
		float nDl = dot(ffnormal, L);

		if (nDl > 0.0f) {
			// cast shadow ray
			PerRayData_shadow shadow_prd;
			shadow_prd.attenuation = make_float3(1.0f);
			float Ldist = length(light.pos - hit_point);
			optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist);
			rtTrace(top_shadower, shadow_ray, shadow_prd);
			float3 light_attenuation = shadow_prd.attenuation;

			if (fmaxf(light_attenuation) > 0.0f) {
				float3 Lc = light.color * light_attenuation;
				color += Kd * nDl * Lc;

				float3 H = normalize(L - ray.direction);
				float nDh = dot(ffnormal, H);
				if (nDh > 0)
					color += Ks * Lc * pow(nDh, phong_exp);
			}

		}
	}
	prd_radiance.result = color;
}


RT_PROGRAM void closest_hit_radiance2()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 color = Ka * ambient_light_color;

	float3 hit_point = ray.origin + t_hit * ray.direction;

	for (int i = 0; i < lights.size(); ++i) {
		BasicLight light = lights[i];
		float3 L = normalize(light.pos - hit_point);
		float nDl = dot(ffnormal, L);

		if (nDl > 0) {
			float3 Lc = light.color;
			color += Kd * nDl * Lc;

			float3 H = normalize(L - ray.direction);
			float nDh = dot(ffnormal, H);
			if (nDh > 0)
				color += Ks * Lc * pow(nDh, phong_exp);

		}
	}
	prd_radiance.result = color;
}



rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float2, barycentrics, attribute barycentrics, );

rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3>   index_buffer;

RT_PROGRAM void triangle_attributes()
{
	const int3   v_idx = index_buffer[rtGetPrimitiveIndex()];
	const float3 v0 = vertex_buffer[v_idx.x];
	const float3 v1 = vertex_buffer[v_idx.y];
	const float3 v2 = vertex_buffer[v_idx.z];
	const float3 Ng = optix::cross(v1 - v0, v2 - v0);

	geometric_normal = optix::normalize(Ng);

	barycentrics = rtGetTriangleBarycentrics();
	texcoord = make_float3(barycentrics.x, barycentrics.y, 0.0f);

	if (normal_buffer.size() == 0)
	{
		shading_normal = geometric_normal;
	}
	else
	{
		shading_normal = normal_buffer[v_idx.y] * barycentrics.x + normal_buffer[v_idx.z] * barycentrics.y
			+ normal_buffer[v_idx.x] * (1.0f - barycentrics.x - barycentrics.y);
	}

	if (texcoord_buffer.size() == 0)
	{
		texcoord = make_float3(0.0f, 0.0f, 0.0f);
	}
	else
	{
		const float2 t0 = texcoord_buffer[v_idx.x];
		const float2 t1 = texcoord_buffer[v_idx.y];
		const float2 t2 = texcoord_buffer[v_idx.z];
		texcoord = make_float3(t1 * barycentrics.x + t2 * barycentrics.y + t0 * (1.0f - barycentrics.x - barycentrics.y));
	}
}