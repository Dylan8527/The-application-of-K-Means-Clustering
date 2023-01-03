#include <SLIC.h>
#include <utils.h>
#include "cmdline.h"

unsigned int getRand() {return rand() % 0xff;}

void ProcessOneImage(string& filename, string& savepath1,string& savepath2, int& m_spcount, double& m_compactness) {
    UINT* img = NULL;
    int width = 0;
    int height = 0;

    // 1. Read image and pre-process image
    LoadARGBImage(filename, img, width, height);

    int sz = width * height;
    int* labels = new int[sz];
    int numlabels(0);
    // 2. Superpixel generation process
    SLIC slic;
    slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img, width, height, labels, numlabels, m_spcount, m_compactness);

    // 3. draw contours ~
    slic.DrawContoursAroundSegments(img, labels, width, height, 0);
    SaveImage(img, width, height, savepath1);

    // 3. draw colors ~
    // new random color values
    unsigned int* color = new unsigned int[m_spcount];
    for (int i = 0; i < m_spcount; i++) {   color[i] = (getRand() | (getRand() << 8) | (getRand() << 16)) % 0xffffff;}
    slic.Drawcolors_Forlabels(img, labels, width, height, color);
    SaveImage(img, width, height, savepath2);

    if(labels) delete [] labels;
    if(img) delete [] img;
}

int main(int argc, char* argv[])
{
    // create a parser 
    cmdline::parser p;
    // define arguments and their short forms
    p.add<string>("input", 'i', "input image path", true, "");
    p.add<int>("spcount", 's', "superpixel count", false, 200);
    p.add<double>("compactness", 'c', "compactness", false, 20, cmdline::range(1.0, 40.0));
    // parse the command line
    p.parse_check(argc, argv);
    string input = p.get<string>("input");
    if (input.empty()) {
        printf("Input path is empty. Please check your input path.\n");
        return -1;
    }
    if (input.find(".JPG") == string::npos) {
        printf("Input path is not a jpegfile. Please check your input path.\n");
        return -1;
    }
    //---------------------------------------------------------
    int m_spcount = p.get<int>("spcount");
    double m_compactness = p.get<double>("compactness");
    //---------------------------------------------------------
    string filename = p.get<string>("input");
    // savepath replace the last path with new jpg nam 
    char param_extension[256];
    std::sprintf(param_extension, "_spcount_%d_compactness_%.1f", m_spcount, m_compactness);
    string param_ext = param_extension;
    string savepath1 = filename.substr(0, filename.find_last_of(".")) + "_contours"+ param_ext + ".jpg";
    // string savepath1 = std::filesystem::path(filename).replace_extension("") + "_contours.jpg"
    string savepath2 = filename.substr(0, filename.find_last_of(".")) + "_colors" + param_ext +".jpg";
    // string savepath2 = 
    ProcessOneImage(filename, savepath1, savepath2, m_spcount, m_compactness);
    printf("Done!");
    return 0;
}
