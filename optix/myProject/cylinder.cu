
#include "tutorial.h"
#include <optixu/optixu_aabb.h>
#define ZERO 1e-20

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, cylinder_boxmin, , );
rtDeclareVariable(float3, cylinder_boxmax, , );

RT_PROGRAM void cylinder_bounds(int primIdx, float result[6])
{
	float3 center = (cylinder_boxmin + cylinder_boxmax) / 2.0f;
	float radius = (cylinder_boxmax.x - cylinder_boxmin.x) / 2.0f;
	optix::Aabb* aabb = (optix::Aabb*)result;
	aabb->m_min = make_float3(center.x - radius, cylinder_boxmin.y, center.z - radius);
	aabb->m_max = make_float3(center.x + radius, cylinder_boxmax.y, center.z + radius);
}


RT_PROGRAM void cylinder_intersect(int primIdx) {
	float3 center = (cylinder_boxmin + cylinder_boxmax) / 2.0;
	float radius = cylinder_boxmax.x - cylinder_boxmin.x;
	radius = radius / 2.0;
	float2 v_xz = make_float2(ray.direction.x, ray.direction.z);
	float2 o_xz = make_float2(ray.origin.x - center.x, ray.origin.z - center.z);
	float3 o = make_float3(ray.origin.x - center.x, ray.origin.y, ray.origin.z - center.z);
	//o = make_float3(0, 0, 0);
	float a = dot(v_xz, v_xz);
	float b = 2 * dot(o_xz, v_xz);
	float c = dot(o_xz, o_xz) - radius * radius;
	float delta = b * b - 4 * a * c;
	float h = cylinder_boxmax.y - cylinder_boxmin.y;
	if (delta > ZERO) {
		bool check_second = true;
		float root1 = (-b - sqrtf(delta)) / (2.0f * a + ZERO);
		float root2 = (-b + sqrtf(delta)) / (2.0f * a + ZERO);
		float root3 = (cylinder_boxmin.y - ray.origin.y) / (ray.direction.y + ZERO);
		float root4 = (cylinder_boxmax.y - ray.origin.y) / (ray.direction.y + ZERO);
		float3 dest1 = ray.origin + root1 * ray.direction;
		float3 dest2 = ray.origin + root2 * ray.direction;
		bool rule1 = true;
		rule1 = dest1.y<cylinder_boxmax.y && dest1.y>cylinder_boxmin.y && dest2.y<cylinder_boxmax.y && dest2.y>cylinder_boxmin.y;
		if (rule1) {
			if (rtPotentialIntersection(root1)) {
				shading_normal = geometric_normal = make_float3(dest1.x - center.x, 0.0f, dest1.z - center.z) ;
				if (rtReportIntersection(0))
					check_second = false;
			}
			if (check_second) {
				if (rtPotentialIntersection(root2)) {
					shading_normal = geometric_normal = make_float3(dest2.x - center.x, 0.0f, dest2.z - center.z) ;
					rtReportIntersection(0);
				}
			}
		}
		else if (dest2.y >= cylinder_boxmax.y && dest1.y<cylinder_boxmax.y && dest1.y>cylinder_boxmin.y) {
			if (rtPotentialIntersection(root1)) {
				shading_normal = geometric_normal = make_float3(dest1.x - center.x, 0.0f, dest1.z - center.z);
				if (rtReportIntersection(0))
					check_second = false;
			}
			if (check_second) {
				if (rtPotentialIntersection(root4)) {
					shading_normal = geometric_normal = make_float3(0, -1, 0);
					rtReportIntersection(0);
				}
			}
		}
		else if (dest1.y >= cylinder_boxmax.y && dest2.y<cylinder_boxmax.y && dest2.y>cylinder_boxmin.y) {
			if (rtPotentialIntersection(root4)) {
				shading_normal = geometric_normal = make_float3(0, 1 + ZERO, 0);
				if (rtReportIntersection(0))
					check_second = false;
			}
			if (check_second) {
				if (rtPotentialIntersection(root2)) {
					shading_normal = geometric_normal = make_float3(dest2.x - center.x, 0.0f, dest2.z - center.z);
					rtReportIntersection(0);
				}
			}
		}
		else if (dest2.y <= cylinder_boxmin.y && dest1.y<cylinder_boxmax.y && dest1.y>cylinder_boxmin.y) {
			if (rtPotentialIntersection(root1)) {
				shading_normal = geometric_normal = make_float3(dest1.x - center.x, 0.0f + ZERO, dest1.z - center.z);
				if (rtReportIntersection(0))
					check_second = false;
			}
			if (check_second) {
				if (rtPotentialIntersection(root3)) {
					shading_normal = geometric_normal = make_float3(0, 1 + ZERO, 0);
					rtReportIntersection(0);
				}
			}
		}
		else if (dest1.y < cylinder_boxmin.y && dest2.y<cylinder_boxmax.y && dest2.y>cylinder_boxmin.y) {
			if (rtPotentialIntersection(root3)) {
				shading_normal = geometric_normal = make_float3(0, -1 + ZERO, 0);
				if (rtReportIntersection(0))
					check_second = false;
			}
			if (check_second) {
				if (rtPotentialIntersection(root2)) {
					shading_normal = geometric_normal = make_float3(dest2.x - center.x, 0.0f + ZERO, dest2.z - center.z);
					rtReportIntersection(0);
				}
			}
		}
		/*else if (dest1.y <= cylinder_boxmin.y && dest2.y >= cylinder_boxmax.y) {
			if (rtPotentialIntersection(root3)) {
				shading_normal = geometric_normal = make_float3(0, -1, 0);
				if (rtReportIntersection(0))
					check_second = false;
			}
			if (check_second) {
				if (rtPotentialIntersection(root4)) {
					shading_normal = geometric_normal = make_float3(0, 1, 0);
					rtReportIntersection(0);
				}
			}
		}
		else if (dest2.y <= cylinder_boxmin.y && dest1.y >= cylinder_boxmax.y) {
			if (rtPotentialIntersection(root4)) {
				shading_normal = geometric_normal = make_float3(0, 1, 0);
				if (rtReportIntersection(0))
					check_second = false;
			}
			if (check_second) {
				if (rtPotentialIntersection(root3)) {
					shading_normal = geometric_normal = make_float3(0, -1, 0);
					rtReportIntersection(0);
				}
			}
		}*/
	}
}
