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

//获得基本频率统计数据
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

//训练隐马模型参数
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
                //因概率过小，所以都乘上100
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
			    //因概率过小，所以都乘上100
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

//使用viterbi算法解码
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
            temp.push_back(0.0);
            //获取词i选词性j的最大价值
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

//判断该行数据是否属于测试集，用于交叉验证
//total为(训练集+测试集)总行数，lineNum为判断行，fold为fold折交叉验证，no为第no次交叉验证
bool judgeTest(int total, int lineNum, int fold, int no)
{
    int testSize = total / fold;
    if((lineNum > testSize * (no - 1)) && (lineNum <= testSize * no))
        return true;
    else
        return false;
}

//用于获取交叉验证所需的数据。由于系统空间有限(2GB硬盘)，所以每次验证分别获取数据，没有一次获取全部
void corpusPreprocess(const string &rawData, const int fold)
{
    ifstream input(rawData);
    ofstream oTestData("testData.txt", ofstream::app);
    ofstream otaggedTrainingData("taggedTrainingData.txt", ofstream::app);
    ofstream ountaggedTrainingData("untaggedTrainingData.txt", ofstream::app);
    ofstream otaggedTestData("taggedTestData.txt", ofstream::app);
    ofstream ountaggedTestData("untaggedTestData.txt", ofstream::app);
    
    int linenumCnt = 1;
    
    string lineStr, text;
    while(getline(input, lineStr))
    {
        if(judgeTest(allLinenum, linenumCnt, fold, 4))  //**注意根据1~fold修改no**
        {
            otaggedTestData << lineStr << endl;
        }
        else
        {
            otaggedTrainingData << lineStr << endl;
        }
        
        istringstream line(lineStr);
        
        while(line >> text)
        {
            texts.push_back(text);
            
            auto pos = text.find("/");
            string phrase = text.substr(0, pos);
            oTestData << phrase << " ";
            if(judgeTest(allLinenum, linenumCnt, fold, 4))  //**注意根据1~fold修改no**
            {
                ountaggedTestData << phrase << " ";
            }
            else
            {
                ountaggedTrainingData << phrase << " ";
            }
            
        }
        oTestData << endl;
        if(judgeTest(allLinenum, linenumCnt, fold, 4))  //**注意根据1~fold修改no**
        {
            ountaggedTestData << endl;
        }
        else
        {
            ountaggedTrainingData << endl;
        }
        ++linenumCnt;
    }
}

//解码以及计算准确率
void calculate()
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

    cout << "Precision = " << cntCorrect * 1.0 / cntTest << endl;
}
