#include <iostream>
#include <fstream>
#include <string.h>
using namespace std;

int main (void)
{
    string myText;
    string oldText;
    
    while (1) 
    {
        // Output the text from the file
        ifstream MyReadFile("file.txt");
        getline (MyReadFile, myText);
        if(myText == oldText)
        {
            continue;
        }
        else
        {
            oldText = myText;
            cout << '\n' << myText;
        }
    }
    
    return 0;
}