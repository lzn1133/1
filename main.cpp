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

// ��ά�����ṹ
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

// ��ά��������
struct Vec2 {
    float u, v;
};

// ��ṹ��֧��v/vt��v/vn/vt��ʽ��
struct Face {
    int v[3];    // ��������
    int vt[3];   // ������������
    int vn[3];   // ��������
    bool hasNormals = false; // ����Ƿ��з�������
};

// ȫ�ֱ���
vector<Vec3> vertices;    // ��������
vector<Vec3> normals;     // �����������ļ��ṩ�����ó���
vector<Vec2> texcoords;   // ��������
vector<Face> faces;       // ������

// ģ�Ͱ�Χ�к�������Ϣ
Vec3 bbox_min = { 1e9, 1e9, 1e9 }, bbox_max = { -1e9, -1e9, -1e9 };
Vec3 model_center = { 0, 0, 0 };
float model_scale = 1.0f;

// ��ͼ���Ʊ���
float rotX = 0, rotY = 0, zoom = 1.0f;
float panX = 0, panY = 0;
int lastX = 0, lastY = 0;
bool leftDown = false, rightDown = false, middleDown = false;

GLuint textureID;  // ����ID

// ������ͼ
void resetView() {
    rotX = rotY = panX = panY = 0;
    zoom = 1.0f;
}

// ���������ݣ�֧��v/vt��v/vn/vt��ʽ��
void parseFace(const vector<string>& tokens) {
    if (tokens.size() < 3) return;  // ������Ҫ3������

    Face f;
    bool hasNormals = false;

    // ����һ���������ݸ�ʽ
    string firstPart = tokens[1];
    size_t slash1 = firstPart.find('/');
    size_t slash2 = firstPart.find('/', slash1 + 1);

    // ȷ����ʽ����
    if (slash2 != string::npos && slash2 > slash1 + 1) {
        hasNormals = true;  // ��ʽΪv/vn/vt��v//vn
    }

    for (int i = 0; i < 3; ++i) {
        string part = tokens[i];
        size_t slash1 = part.find('/');
        size_t slash2 = part.find('/', slash1 + 1);

        if (slash1 == string::npos) continue;

        // ������������
        f.v[i] = stoi(part.substr(0, slash1)) - 1;

        // ����������������
        if (slash2 != string::npos) {
            // v/vt/vn ��ʽ
            f.vt[i] = stoi(part.substr(slash1 + 1, slash2 - slash1 - 1)) - 1;
            f.vn[i] = stoi(part.substr(slash2 + 1)) - 1;
        }
        else if (part.length() > slash1 + 1) {
            // v/vt ��ʽ
            f.vt[i] = stoi(part.substr(slash1 + 1)) - 1;
            f.vn[i] = -1;  // ���Ϊ������
        }
        else {
            // v//vn ��ʽ
            f.vt[i] = -1;  // ���Ϊ������
            f.vn[i] = stoi(part.substr(slash2 + 1)) - 1;
        }
    }

    f.hasNormals = hasNormals;
    faces.push_back(f);
}

// ���㷨�ߣ���OBJ�ļ���û���ṩ����ʱ��
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

        // �����߼ӵ�ÿ��������
        for (int i = 0; i < 3; ++i) {
            normals[face.v[i]] = normals[face.v[i]] + normal;
        }
    }

    // ��һ�����з���
    for (auto& n : normals) {
        n.normalize();
    }
}

// ����Ƿ���Ҫ���㷨��
bool needCalculateNormals() {
    // ����Ƿ����������������
    for (const auto& face : faces) {
        if (face.hasNormals) return false;
    }
    return true;
}

// ����OBJ�ļ�
void loadOBJ(const string& filename) {
    ifstream file(filename);
    if (!file) {
        cerr << "��ʧ��: " << filename << endl;
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

            // ���°�Χ��
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

    // �����Ҫ�����㷨��
    if (needCalculateNormals()) {
        calculateNormals();
        cout << "����: OBJ�ļ��������������ݣ����Զ�����ƽ������" << endl;
    }

    // ����ģ�����ĺ����ű���
    model_center = {
        (bbox_min.x + bbox_max.x) / 2.0f,
        (bbox_min.y + bbox_max.y) / 2.0f,
        (bbox_min.z + bbox_max.z) / 2.0f
    };

    float scaleX = bbox_max.x - bbox_min.x;
    float scaleY = bbox_max.y - bbox_min.y;
    float scaleZ = bbox_max.z - bbox_min.z;
    model_scale = 2.0f / max({ scaleX, scaleY, scaleZ });

    cout << "ģ�ͼ��سɹ�: ����=" << vertices.size()
        << " ����=" << normals.size()
        << " ��������=" << texcoords.size()
        << " ��=" << faces.size() << endl;
}

// ��������
void loadTexture(const string& filename) {
    Mat img = imread(filename, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "�޷���������: " << filename << endl;
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

// ��ʼ������
void initLighting() {
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat pos[] = { 0, 0, 5, 1 };
    glLightfv(GL_LIGHT0, GL_POSITION, pos);
    GLfloat white[] = { 1, 1, 1, 1 };
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, white);
}

// ����ģ��
void drawModel() {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glBegin(GL_TRIANGLES);
    for (const auto& f : faces) {
        for (int i = 0; i < 3; ++i) {
            // ʹ���ļ��еķ��߻����ķ���
            Vec3 n = f.hasNormals ? normals[f.vn[i]] : normals[f.v[i]];

            // �������꣨����У�
            if (f.vt[i] >= 0 && f.vt[i] < texcoords.size()) {
                Vec2 t = texcoords[f.vt[i]];
                glTexCoord2f(t.u, t.v);
            }

            // ��������
            Vec3 v = vertices[f.v[i]];

            glNormal3f(n.x, n.y, n.z);
            glVertex3f(v.x, v.y, v.z);
        }
    }
    glEnd();

    glDisable(GL_TEXTURE_2D);
}

// ��ʾ����
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

// ���ڴ�С�ı�ص�
void reshape(int w, int h) {
    if (h == 0) h = 1;
    float aspect = (float)w / h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, aspect, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

// ���ص�����
void mouse(int button, int state, int x, int y) {
    lastX = x; lastY = y;
    if (button == GLUT_LEFT_BUTTON) leftDown = (state == GLUT_DOWN);
    if (button == GLUT_RIGHT_BUTTON) rightDown = (state == GLUT_DOWN);
    if (button == GLUT_MIDDLE_BUTTON) middleDown = (state == GLUT_DOWN);
}

// ����ƶ��ص�
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

// ����������
void mouseWheel(int wheel, int direction, int x, int y) {
    if (direction > 0) {
        zoom *= 1.1f; // �Ŵ�
    }
    else {
        zoom /= 1.1f; // ��С
    }
    glutPostRedisplay();
}

// ���̻ص�
void keyboard(unsigned char key, int, int) {
    if (key == 32) resetView();  // �ո��������ͼ
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    char szCommandLine[] = "python demos/demo_reconstruct.py -i TestSamples/examples --saveDepth True --saveObj True";
    USES_CONVERSION;
    LPWSTR str1 = A2W(szCommandLine);

    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    si.dwFlags = STARTF_USESHOWWINDOW; //�ƶ�wShowWindow��Ա
    si.wShowWindow = TRUE; //Ϊ�棬��ʾ���̵�������

    BOOL bRet = ::CreateProcess(
        NULL,//���ڴ�ָ����ִ���ļ����ļ���
        str1, //�����в���
        NULL,//Ĭ�Ͻ��̵İ�ȫ��
        NULL,//Ĭ���̵߳İ�ȫ��
        FALSE,//ָ����ǰ�����ڵľ�������Ա��ӽ��̼̳�
        CREATE_NEW_CONSOLE,//Ϊ�½��̴���һ���µĿ���̨����
        NULL,//ʹ�ñ����̵Ļ�������
        NULL,//ʹ�ñ����̵���������Ŀ¼
        &si,
        &pi);
    // ����OBJģ��
    loadOBJ("TTT.obj");

    // ��ʼ��GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(900, 700);
    glutCreateWindow("ͨ��OBJ�鿴�� (֧��v/vt��v/vn/vt��ʽ)");

    // ��ʼ��OpenGL״̬
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    initLighting();

    // ��������
    loadTexture("TTT.png");

    // ���ûص�����
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMouseWheelFunc(mouseWheel);
    glutKeyboardFunc(keyboard);

    // ������ѭ��
    glutMainLoop();
    return 0;
}