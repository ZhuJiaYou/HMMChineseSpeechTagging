using namespace std;

//语料数据
vector<string> texts;  //训练样本中词语和词性
vector<string> phrases;  //训练样本中不带标注词语
vector<string> characters;  //训练样本没有词语的词性序列
vector<string> diffPhras;  //语料中各不相同的词组
vector<string> diffChars;  //语料中各不相同的词性符号

//频率统计数据
map<string, int> emissonFrequencyMap;  //词与其标记词性的频率
map<string, int> transformFrequencyMap;  //连续出现的两个词性及其频率
map<string, int> phrasesMap;  //词组及其频率
map<string, int> charactersMap;  //词性及其频率
int charactersNum = 0;
int phraseNum = 0;
map<string, int> phrasePosition;
int allLinenum = 0;

//HMM参数
vector<double> prioriProbability;  //词性的先验概率
vector<vector<double>> transformProbability;  //词性的转移验概率矩阵
vector<vector<double>> emissionProbability;  //发射矩阵

void readCorpus(const string file)
{
    ifstream input(file);
//    ofstream output1("emisson.txt", ofstream::app);
//    ofstream output2("transform.txt", ofstream::app);
//    ofstream output3("phrase.txt", ofstream::app);
//    ofstream output4("character.txt", ofstream::app);
    
    string lineStr, text;
    while(getline(input, lineStr))
    {
        ++allLinenum;
        istringstream line(lineStr);
        
        string preCharacter;
        bool first = true;
        while(line >> text)
        {
            texts.push_back(text);
            ++emissonFrequencyMap[text];
            
            auto pos = text.find("/");
            string phrase = text.substr(0, pos);
            string character = text.substr(pos + 1, text.size() - (pos + 1));
            
            phrases.push_back(phrase);
            ++phrasesMap[phrase];
            
            characters.push_back(character);
            ++charactersMap[character];
            
            if(!first)
                ++transformFrequencyMap[(preCharacter + "," + character)];
            
            first = false;
            preCharacter = character;
        }
    }
    
    //统计存储不同的词
    int pos = 0;
    for(auto val : phrasesMap)
    {
        diffPhras.push_back(val.first);
        phrasePosition[val.first] = pos++;
    }
    phraseNum = diffPhras.size();
    
    //统计存储不同的词性
    for(auto val : charactersMap)
        diffChars.push_back(val.first);
    charactersNum = diffChars.size();
}

void getHmmParameters()
{
    //计算词性先验概率
    auto allCharacterCount = characters.size();
    for(auto val : diffChars)
        prioriProbability.push_back(charactersMap[val] * 1.0 / allCharacterCount);
    
    //计算词性转移概率
    for(decltype(charactersNum) i = 0; i < charactersNum; ++i)
    {
        vector<double> itrans;
        for(decltype(charactersNum) j = 0; j < charactersNum; ++j)
        {
            string front = diffChars[i];
            string last = diffChars[j];
            string trans = front + "," + last;
            
            if(transformFrequencyMap.find(trans) != transformFrequencyMap.end())
            {
                itrans.push_back(transformFrequencyMap[trans] * 100.0 / charactersMap[front]);
            }
            else
            {
                itrans.push_back(0.0);
            }
        }
        transformProbability.push_back(itrans);
    }
        
    //计算发射概率
    for(decltype(charactersNum) i = 0; i < charactersNum; ++i)
    {
        vector<double> iemt;
        for(decltype(phraseNum) j = 0; j < phraseNum; ++j)
        {
            string chars = diffChars[i];
			string phras = diffPhras[j];
			string text = phras + "/" + chars;

			if(emissonFrequencyMap.find(text) != emissonFrequencyMap.end())
			{
				iemt.push_back(emissonFrequencyMap[text] * 100.0 / charactersMap[chars]);
            }
            else
            {
                iemt.push_back(0.0);
            }
        }
        emissionProbability.push_back(iemt);
    }
}

void viterbi(vector<string> &v)
{
    
    vector<vector<double>> value;  //词组i选择词性j的最大价值
    vector<vector<int>> previous;  //value取得最大值时的前驱
    int position;
    
    vector<int> preTemp0;
    for(decltype(charactersNum) j = 0; j < charactersNum; ++j)
        preTemp0.push_back(0);
    previous.push_back(preTemp0);
    
    //初始化第一个字在不同词性下的value，是其发射概率
    if(phrasePosition.find(v[0]) != phrasePosition.end())
    {
        position = phrasePosition[v[0]];
        vector<double> temp;
        for(decltype(charactersNum) j = 0; j < charactersNum; ++j)
            temp.push_back(prioriProbability[j] * emissionProbability[j][position]);
        value.push_back(temp);
    }
    else
    {
        vector<double> temp;
        for(decltype(charactersNum) j = 0; j < charactersNum; ++j)
            temp.push_back(1.0);
        value.push_back(temp);
    }

    
    for(decltype(v.size()) i = 1; i < v.size(); ++i)
    {
        vector<double> temp;
        vector<int> preTemp;
        if(phrasePosition.find(v[i]) == phrasePosition.end())
        {
            for(decltype(charactersNum) j = 0; j < charactersNum; ++j)
                temp.push_back(1.0);
            value.push_back(temp);
            continue;
        }
        
        position = phrasePosition[v[i]];
        for(decltype(charactersNum) j = 0; j < charactersNum; ++j)
        {
            double max = value[i - 1][0] * transformProbability[0][j] * emissionProbability[j][position];
            
            int index = 0;
            //获取词i选词性j的最大价值
            temp.push_back(0.0);
            for(decltype(charactersNum) k = 1; k < charactersNum; ++k)  //前一个词可能的词性
            {
                temp[j] = (value[i - 1][k] * transformProbability[k][j] * emissionProbability[j][position]);
                if(temp[j] > max)
                {
                    index = k;
                    max = temp[j];
                }
            }
            temp[j] = max;
            preTemp.push_back(index);
        }
        value.push_back(temp);
        previous.push_back(preTemp);
    }
    
    //首先找到最后一个字在不同词性下的最大值，然后由此最大值回溯查找其他最大值
    double max = -1.0;
    int index = 0;
    for(decltype(charactersNum) i = 0; i < charactersNum; ++i)
    {
        if(max < value[v.size() - 1][i])
        {
            index = i;
            max = value[v.size() - 1][i];
        }
    }

    for(int i = v.size() - 1; i >= 0; --i)
    {
        v[i] = v[i] + "/" + diffChars[index];
        index = previous[i][index];
    }
}

void corpusPreprocess(const string &rawData, const double partRatio)
{
    ifstream input(rawData);
    ofstream oTestData("testData.txt", ofstream::app);
    ofstream otaggedTrainingData("taggedTrainingData.txt", ofstream::app);
    ofstream ountaggedTrainingData("untaggedTrainingData", ofstream::app);
    ofstream otaggedTestData("taggedTestData.txt", ofstream::app);
    ofstream ountaggedTestData("untaggedTestData.txt", ofstream::app);
    
    int breakpoint = static_cast<int>(partRatio * allLinenum);
    int linenumCnt = 1;
    
    //vector<string> words;
    string lineStr, text;
    while(getline(input, lineStr))
    {
        if(linenumCnt <= breakpoint)
        {
            otaggedTrainingData << lineStr << endl;
        }
        else
        {
            otaggedTestData << lineStr << endl;
        }
        
        istringstream line(lineStr);
        
        while(line >> text)
        {
            texts.push_back(text);
            
            auto pos = text.find("/");
            string phrase = text.substr(0, pos);
            oTestData << phrase << " ";
            if(linenumCnt <= breakpoint)
            {
                ountaggedTrainingData << phrase << " ";
            }
            else
            {
                ountaggedTestData << phrase << " ";
            }
            
        }
        oTestData << endl;
        if(linenumCnt <= breakpoint)
        {
            ountaggedTrainingData << endl;
        }
        else
        {
            ountaggedTestData << endl;
        }
        ++linenumCnt;
    }
}

double calculate()
{
    int cntCorrect = 0;
    int cntTest = 0;
    ifstream inputTagged("taggedTestData.txt");
    ifstream inputUntagged("untaggedTestData.txt");

    string untaggedLineStr, taggedLineStr;
    while(getline(inputUntagged, untaggedLineStr))
    {
        getline(inputTagged, taggedLineStr);
        
        istringstream untaggedLine(untaggedLineStr);
        istringstream taggedLine(taggedLineStr);
        
        string processedText, originText;
        vector<string> processedSentence, originSetence;
        while(untaggedLine >> processedText)
        {
            taggedLine >> originText;
            processedSentence.push_back(processedText);
            originSetence.push_back(originText);
        }

        viterbi(processedSentence);

        for(int i = 0; i < processedSentence.size(); ++i)
        {
            ++cntTest;
            if(processedSentence[i] == originSetence[i])
                ++cntCorrect;
        }
    }

    return (cntCorrect * 1.0 / cntTest);
}
