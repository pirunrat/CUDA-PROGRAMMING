#include <iostream>


int main(){
    int N = 3;

    size_t size = N * sizeof(float);
    
    std::cout << "Size of array with " << N << "Float elements: " << size << "Bytes" << std::endl;

    return 0;
}