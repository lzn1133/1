#include <GL/freeglut.h>
#include <iostream>
#include <atlconv.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 三维向量结构
struct Vec3 {
    float x, y, z;
    Vec3 operator-(const Vec3& rhs) const {
        return { x - rhs.x, y - rhs.y, z - rhs.z };
    }
    Vec3 operator+(const Vec3& rhs) const {
        return { x + rhs.x, y + rhs.y, z + rhs.z };
    }
    Vec3 cross(const Vec3& rhs) const {
        return {
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        };
    }
    void normalize() {
        float len = sqrt(x * x + y * y + z * z);
        if (len > 0) {
            x /= len; y /= len; z /= len;
        }
    }
};

// 二维纹理坐标
struct Vec2 {
    float u, v;
};

// 面结构（支持v/vt和v/vn/vt格式）
struct Face {
    int v[3];    // 顶点索引
    int vt[3];   // 纹理坐标索引
    int vn[3];   // 法线索引
    bool hasNormals = false; // 标记是否有法线数据
};

// 全局变量
vector<Vec3> vertices;    // 顶点坐标
vector<Vec3> normals;     // 法线向量（文件提供或计算得出）
vector<Vec2> texcoords;   // 纹理坐标
vector<Face> faces;       // 面数据

// 模型包围盒和缩放信息
Vec3 bbox_min = { 1e9, 1e9, 1e9 }, bbox_max = { -1e9, -1e9, -1e9 };
Vec3 model_center = { 0, 0, 0 };
float model_scale = 1.0f;

// 视图控制变量
float rotX = 0, rotY = 0, zoom = 1.0f;
float panX = 0, panY = 0;
int lastX = 0, lastY = 0;
bool leftDown = false, rightDown = false, middleDown = false;

GLuint textureID;  // 纹理ID

// 重置视图
void resetView() {
    rotX = rotY = panX = panY = 0;
    zoom = 1.0f;
}

// 解析面数据（支持v/vt和v/vn/vt格式）
void parseFace(const vector<string>& tokens) {
    if (tokens.size() < 3) return;  // 至少需要3个顶点

    Face f;
    bool hasNormals = false;

    // 检查第一个顶点数据格式
    string firstPart = tokens[1];
    size_t slash1 = firstPart.find('/');
    size_t slash2 = firstPart.find('/', slash1 + 1);

    // 确定格式类型
    if (slash2 != string::npos && slash2 > slash1 + 1) {
        hasNormals = true;  // 格式为v/vn/vt或v//vn
    }

    for (int i = 0; i < 3; ++i) {
        string part = tokens[i];
        size_t slash1 = part.find('/');
        size_t slash2 = part.find('/', slash1 + 1);

        if (slash1 == string::npos) continue;

        // 解析顶点索引
        f.v[i] = stoi(part.substr(0, slash1)) - 1;

        // 解析纹理坐标索引
        if (slash2 != string::npos) {
            // v/vt/vn 格式
            f.vt[i] = stoi(part.substr(slash1 + 1, slash2 - slash1 - 1)) - 1;
            f.vn[i] = stoi(part.substr(slash2 + 1)) - 1;
        }
        else if (part.length() > slash1 + 1) {
            // v/vt 格式
            f.vt[i] = stoi(part.substr(slash1 + 1)) - 1;
            f.vn[i] = -1;  // 标记为无数据
        }
        else {
            // v//vn 格式
            f.vt[i] = -1;  // 标记为无数据
            f.vn[i] = stoi(part.substr(slash2 + 1)) - 1;
        }
    }

    f.hasNormals = hasNormals;
    faces.push_back(f);
}

// 计算法线（当OBJ文件中没有提供法线时）
void calculateNormals() {
    normals.resize(vertices.size(), { 0,0,0 });

    for (const auto& face : faces) {
        Vec3 v0 = vertices[face.v[0]];
        Vec3 v1 = vertices[face.v[1]];
        Vec3 v2 = vertices[face.v[2]];

        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 normal = edge1.cross(edge2);
        normal.normalize();

        // 将法线加到每个顶点上
        for (int i = 0; i < 3; ++i) {
            normals[face.v[i]] = normals[face.v[i]] + normal;
        }
    }

    // 归一化所有法线
    for (auto& n : normals) {
        n.normalize();
    }
}

// 检查是否需要计算法线
bool needCalculateNormals() {
    // 检查是否有面包含法线数据
    for (const auto& face : faces) {
        if (face.hasNormals) return false;
    }
    return true;
}

// 加载OBJ文件
void loadOBJ(const string& filename) {
    ifstream file(filename);
    if (!file) {
        cerr << "打开失败: " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        istringstream ss(line);
        string word;
        ss >> word;

        if (word == "v") {
            Vec3 v;
            ss >> v.x >> v.y >> v.z;
            vertices.push_back(v);

            // 更新包围盒
            bbox_min.x = min(bbox_min.x, v.x);
            bbox_min.y = min(bbox_min.y, v.y);
            bbox_min.z = min(bbox_min.z, v.z);
            bbox_max.x = max(bbox_max.x, v.x);
            bbox_max.y = max(bbox_max.y, v.y);
            bbox_max.z = max(bbox_max.z, v.z);
        }
        else if (word == "vn") {
            Vec3 n;
            ss >> n.x >> n.y >> n.z;
            normals.push_back(n);
        }
        else if (word == "vt") {
            Vec2 t;
            ss >> t.u >> t.v;
            texcoords.push_back(t);
        }
        else if (word == "f") {
            vector<string> tokens;
            string token;
            while (ss >> token) tokens.push_back(token);
            parseFace(tokens);
        }
    }

    // 如果需要，计算法线
    if (needCalculateNormals()) {
        calculateNormals();
        cout << "警告: OBJ文件不包含法线数据，已自动计算平滑法线" << endl;
    }

    // 计算模型中心和缩放比例
    model_center = {
        (bbox_min.x + bbox_max.x) / 2.0f,
        (bbox_min.y + bbox_max.y) / 2.0f,
        (bbox_min.z + bbox_max.z) / 2.0f
    };

    float scaleX = bbox_max.x - bbox_min.x;
    float scaleY = bbox_max.y - bbox_min.y;
    float scaleZ = bbox_max.z - bbox_min.z;
    model_scale = 2.0f / max({ scaleX, scaleY, scaleZ });

    cout << "模型加载成功: 顶点=" << vertices.size()
        << " 法线=" << normals.size()
        << " 纹理坐标=" << texcoords.size()
        << " 面=" << faces.size() << endl;
}

// 加载纹理
void loadTexture(const string& filename) {
    Mat img = imread(filename, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "无法加载纹理: " << filename << endl;
        return;
    }

    cvtColor(img, img, COLOR_BGR2RGB);
    flip(img, img, 0);

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows,
        0, GL_RGB, GL_UNSIGNED_BYTE, img.data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

// 初始化光照
void initLighting() {
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat pos[] = { 0, 0, 5, 1 };
    glLightfv(GL_LIGHT0, GL_POSITION, pos);
    GLfloat white[] = { 1, 1, 1, 1 };
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, white);
}

// 绘制模型
void drawModel() {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glBegin(GL_TRIANGLES);
    for (const auto& f : faces) {
        for (int i = 0; i < 3; ++i) {
            // 使用文件中的法线或计算的法线
            Vec3 n = f.hasNormals ? normals[f.vn[i]] : normals[f.v[i]];

            // 纹理坐标（如果有）
            if (f.vt[i] >= 0 && f.vt[i] < texcoords.size()) {
                Vec2 t = texcoords[f.vt[i]];
                glTexCoord2f(t.u, t.v);
            }

            // 顶点坐标
            Vec3 v = vertices[f.v[i]];

            glNormal3f(n.x, n.y, n.z);
            glVertex3f(v.x, v.y, v.z);
        }
    }
    glEnd();

    glDisable(GL_TEXTURE_2D);
}

// 显示函数
void display() {
    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt(0, 0, 3 / zoom, 0, 0, 0, 0, 1, 0);

    glTranslatef(panX, panY, 0);
    glRotatef(rotY, 1, 0, 0);
    glRotatef(rotX, 0, 1, 0);
    glScalef(model_scale, model_scale, model_scale);
    glTranslatef(-model_center.x, -model_center.y, -model_center.z);

    drawModel();
    glutSwapBuffers();
}

// 窗口大小改变回调
void reshape(int w, int h) {
    if (h == 0) h = 1;
    float aspect = (float)w / h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, aspect, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

// 鼠标回调函数
void mouse(int button, int state, int x, int y) {
    lastX = x; lastY = y;
    if (button == GLUT_LEFT_BUTTON) leftDown = (state == GLUT_DOWN);
    if (button == GLUT_RIGHT_BUTTON) rightDown = (state == GLUT_DOWN);
    if (button == GLUT_MIDDLE_BUTTON) middleDown = (state == GLUT_DOWN);
}

// 鼠标移动回调
void motion(int x, int y) {
    int dx = x - lastX, dy = y - lastY;
    if (leftDown) {
        rotX += dx * 0.5f;
        rotY += dy * 0.5f;
    }
    else if (rightDown) {
        panX += dx * 0.01f;
        panY -= dy * 0.01f;
    }
    else if (middleDown) {
        zoom *= 1.0f + dy * 0.01f;
    }
    lastX = x; lastY = y;
    glutPostRedisplay();
}

// 鼠标滚轮缩放
void mouseWheel(int wheel, int direction, int x, int y) {
    if (direction > 0) {
        zoom *= 1.1f; // 放大
    }
    else {
        zoom /= 1.1f; // 缩小
    }
    glutPostRedisplay();
}

// 键盘回调
void keyboard(unsigned char key, int, int) {
    if (key == 32) resetView();  // 空格键重置视图
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    char szCommandLine[] = "python demos/demo_reconstruct.py -i TestSamples/examples --saveDepth True --saveObj True";
    USES_CONVERSION;
    LPWSTR str1 = A2W(szCommandLine);

    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    si.dwFlags = STARTF_USESHOWWINDOW; //制定wShowWindow成员
    si.wShowWindow = TRUE; //为真，显示进程的主窗口

    BOOL bRet = ::CreateProcess(
        NULL,//不在此指定可执行文件的文件名
        str1, //命令行参数
        NULL,//默认进程的安全性
        NULL,//默认线程的安全性
        FALSE,//指定当前进程内的句柄不可以被子进程继承
        CREATE_NEW_CONSOLE,//为新进程创建一个新的控制台窗口
        NULL,//使用本进程的环境变量
        NULL,//使用本进程的驱动器和目录
        &si,
        &pi);
    // 加载OBJ模型
    loadOBJ("TTT.obj");

    // 初始化GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(900, 700);
    glutCreateWindow("通用OBJ查看器 (支持v/vt和v/vn/vt格式)");

    // 初始化OpenGL状态
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    initLighting();

    // 加载纹理
    loadTexture("TTT.png");

    // 设置回调函数
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMouseWheelFunc(mouseWheel);
    glutKeyboardFunc(keyboard);

    // 进入主循环
    glutMainLoop();
    return 0;
}