#ifndef _SLIC_H_
#define _SLIC_H_
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class SLIC  
{
public:
	SLIC();
	virtual ~SLIC();
    //============================================================================
	// Superpixel segmentation for a given number of superpixels
	//============================================================================
        void DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(
        const unsigned int*                             ubuff,
		const int					width,
		const int					height,
		int*&						klabels,
		int&						numlabels,
                const int&					K,//required number of superpixels
                const double&                                   compactness);//10-20 is a good value for CIELAB space
    //============================================================================
	// 1. sRGB to CIELAB conversion for 2-D images: sRGB -> XYZ -> LAB
	//============================================================================
	void DoRGBtoLABConversion(
		const unsigned int*&		ubuff,
		double*&					lvec,
		double*&					avec,
		double*&					bvec);
                                        //============================================================================
                                        // sRGB to XYZ conversion; helper for RGB2LAB()
                                        //============================================================================
                                        void RGB2XYZ(
                                            const int&					sR,
                                            const int&					sG,
                                            const int&					sB,
                                            double&						X,
                                            double&						Y,
                                            double&						Z);
                                        //============================================================================
                                        // sRGB to CIELAB conversion (uses RGB2XYZ function)
                                        //============================================================================
                                        void RGB2LAB(
                                            const int&					sR,
                                            const int&					sG,
                                            const int&					sB,
                                            double&						lval,
                                            double&						aval,
                                            double&						bval);
    //============================================================================
	// 2. Pick seeds for superpixels when step size of superpixels is given.
	//============================================================================
	void GetLABXYSeeds_ForGivenStepSize(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		const int&					STEP);
    //============================================================================
	// 3. The main SLIC algorithm for generating superpixels
	//============================================================================
	void PerformSuperpixelSLIC(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		int*&						klabels,
		const int&					STEP,
		const double&				m = 10.0);
    //============================================================================
	// 4. Post-processing of SLIC segmentation, to avoid stray labels.
	//============================================================================
	void EnforceLabelConnectivity(
		const int*					labels,
		const int					width,
		const int					height,
		int*&						nlabels,//input labels that need to be corrected to remove stray labels
		int&						numlabels,//the number of labels changes in the end if segments are removed
		const int&					K); //the number of superpixels desired by the user
    //============================================================================
	// 5. Function to draw boundaries around superpixels of a given 'color'.
	// Can also be used to draw boundaries around supervoxels, i.e layer by layer.
	//============================================================================
	void DrawContoursAroundSegments(
		unsigned int*&				segmentedImage,
		int*&						labels,
		const int&					width,
		const int&					height,
		const unsigned int&			color );

	void Drawcolors_Forlabels(
		unsigned int*&				coloredImage,
		int*&						labels,
		const int&					width,
		const int&					height,
		unsigned int*&			color );
private:
	int										m_width;
	int										m_height;
	int										m_depth;

	double*									m_lvec;
	double*									m_avec;
	double*									m_bvec;

	double**								m_lvecvec;
	double**								m_avecvec;
	double**								m_bvecvec;
};
#endif // _SLIC_H_INCLUDED_