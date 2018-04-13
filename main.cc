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
    corpusPreprocess("./data/trainingData.txt", 0.8);
    cout << calculate() << endl;
    
    
    return 0;
}