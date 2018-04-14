#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <memory>
#include "model.h"


int main()
{
    readCorpus("./data/trainingData.txt");
    getHmmParameters();
    corpusPreprocess("./data/trainingData.txt", 5);
    calculate();
    
    
    return 0;
}