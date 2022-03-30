# Project

### UPDATE LINE

* update optix simulation source code
* update refined screenshots(processed by python)
* refined screenshots dataset is on http://47.106.34.252/refined
<!-- * ![dataset](http://47.106.34.252/refinedSet.png) -->
* implementate first-version of SART algorithm
	* assumption:
		* no environmental relexion
		* no container geometric calibration
		* simple objects
	* result
		* 200 times iterations with lambda 0.05:
	
	![sample-0](http://pangpang.live/snapshot-200times-version-0-1.png)
	![sample-1](http://pangpang.live/snapshot-200times-version-0.png)
	![error](http://pangpang.live/lambda005.png)

		* 400 times iterations with lambda 0.025:

	![sample-0](http://pangpang.live/snapshot-400times-version-0-lambda-0025-1.png)
	![sample-1](http://pangpang.live/snapshot-400times-version-0-lambda-0025.png)
	![error](http://pangpang.live/lambda0025.png)

		* 500 times iterations with lambda 0.01:

	![sample-0](http://pangpang.live/snapshot-500times-version-0-lambda-001-1.png)
	![sample-1](http://pangpang.live/snapshot-500times-version-0-lambda-001.png)
	![error](http://pangpang.live/lambda001.png)

### NEXT TODO

* Improvement of SART algorithm implementation


> 3D reconstruction using tomographic method
