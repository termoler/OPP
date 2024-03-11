#include <iostream>
#include <vector>

//#define N 40000
int N = 10000;

double getNorma(std::vector<double>& value) {
    double norma = 0;
    for(double i : value) norma += i * i;
    return norma;
}

std::vector<double> mulMatrixOnVector(const std::vector<double>& matrix, std::vector<double>& y1){
    int sizeMatrix = (int)(matrix.size()/N);
    std::vector<double> x1y1(sizeMatrix, 0);
    for(int i = 0; i < sizeMatrix; i++){
        for(int j = 0; j < N; j++){
            x1y1[i] += matrix[i * N + j] * y1[j];
        }
    }
    return x1y1;
}
std::vector<double> differenceVectors(std::vector<double>& x1, std::vector<double>& y1){
    int sizeX1 = (int)x1.size();
    for(int i = 0; i < sizeX1 ; i++) x1[i] -= y1[i];
    return x1;
}

bool conditionStop(std::vector<double>& xNext, std::vector<double>& b){
    double epsilon = 0.00001;
    return getNorma(xNext) < epsilon * epsilon * getNorma(b);
}


std::vector<double> getDecision(const std::vector<double>& matrix, std::vector<double>& x,
                                std::vector<double>& b) {
    int sizeMatrix = (int)(matrix.size()/N);
    std::vector<double> xNext(N, 0);
    bool flag = false;
    double tay = 0.00001;
    while (!flag) {
        xNext = mulMatrixOnVector(matrix, x);
        xNext = differenceVectors(xNext, b);
        flag = conditionStop(xNext, b);
        for (int i = 0; i < sizeMatrix; i++) {
            xNext[i] = x[i] - tay * xNext[i];
        }
        x = xNext;
    }

    return x;
}

int main() {
    std::vector<double> matrix(N*N, 1);
    std::vector<double> b(N, N + 1);
    std::vector<double> x(N, 0);

    for(int j = 0; j < (int)matrix.size() / N; j ++) {
        matrix[j * N + j] = 2;
    }

    std::vector<double> decision = getDecision(matrix, x, b);
    std::cout << decision[0] << " " << decision[decision.size() - 1];


    return 0;
}
