'use strict';

let gl;                         // The webgl context.
let surface;                    // A surface model
let shProgram;                  // A shader program
let spaceball;                  // A SimpleRotator object that lets the user rotate the view by mouse.

// Vertex shader
const vertexShaderSource = `
attribute vec3 vertex;
attribute vec3 normal;
uniform mat4 ModelViewProjectionMatrix;
varying vec3 norm;
varying vec3 vert;

void main() {
    norm = normal;
    vert = mat3(ModelViewProjectionMatrix) * vertex;
    gl_Position = ModelViewProjectionMatrix * vec4(vertex,1.0);
}`;


// Fragment shader
const fragmentShaderSource = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
   precision highp float;
#else
   precision mediump float;
#endif

varying vec3 norm;
varying vec3 vert;
uniform vec4 color;
uniform vec3 dir;
uniform vec3 pos;
uniform float range, focus;
void main() {
    vec3 toLight = normalize(pos-vert);
    vec3 toView = normalize(-vert);
    vec3 halfVector = normalize(toLight + toView);
    vec3 N = normalize(norm);
    float dotFromDirection = dot(toLight, 
        -dir);
    float inLight = smoothstep(range,range+focus, dotFromDirection);
    float L = inLight * dot(N, toLight);
    float specular = inLight * pow(dot(N, halfVector), 150.0);
    vec3 clr = color.rgb*L+specular;
    gl_FragColor = vec4(clr,1.0);
}`;

const KleinBottle = (u, v, center = [0, 0, 0]) => {
    let x = 2 / 15 * (3 + 5 * Math.cos(u) * Math.sin(u)) * Math.sin(v);
    let y = -1 / 15 * Math.sin(u) * (3 * Math.cos(v) - 3 * Math.pow(Math.cos(u), 2) * Math.cos(v) -
        48 * Math.pow(Math.cos(u), 4) * Math.cos(v) + 48 * Math.pow(Math.cos(u), 6) * Math.cos(v) -
        60 * Math.sin(u) + 5 * Math.cos(u) * Math.cos(v) * Math.sin(u) -
        5 * Math.pow(Math.cos(u), 3) * Math.cos(v) * Math.sin(u) -
        80 * Math.pow(Math.cos(u), 5) * Math.cos(v) * Math.sin(u) +
        80 * Math.pow(Math.cos(u), 7) * Math.cos(v) * Math.sin(u));
    let z = -2 / 15 * Math.cos(u) * (3 * Math.cos(v) - 30 * Math.sin(u) +
        90 * Math.pow(Math.cos(u), 4) * Math.sin(u) - 60 * Math.pow(Math.cos(u), 6) * Math.sin(u) +
        5 * Math.cos(u) * Math.cos(v) * Math.sin(u));

    return glMatrix.vec3.fromValues(x + center[0], y + center[1], z + center[2]);
};

const ParametricSurfaceData = (f, umin, umax, vmin, vmax, nu, nv,
                               xmin, xmax, zmin, zmax, scale = 1, scaley = 0, center = [0, 0, 0]) => {
    const du = (umax - umin) / (nu - 1);
    const dv = (vmax - vmin) / (nv - 1);
    let pts = [];
    let u, v;
    let pt;
    let ymin1 = 0, ymax1 = 0;

    for (let i = 0; i < nu; i++) {
        u = umin + i * du;
        let pt1 = [];
        for (let j = 0; j < nv; j++) {
            v = vmin + j * dv;
            pt = f(u, v, center);
            ymin1 = (pt[1] < ymin1) ? pt[1] : ymin1;
            ymax1 = (pt[1] > ymax1) ? pt[1] : ymax1;
            pt1.push(pt);

        }
        pts.push(pt1);

    }

    const ymin = ymin1 - scaley * (ymax1 - ymin1);
    const ymax = ymax1 + scaley * (ymax1 - ymin1);

    for (let i = 0; i < nu; i++) {
        for (let j = 0; j < nv; j++) {
            pts[i][j] = NormalizePoint(pts[i][j], xmin, xmax, ymin, ymax, zmin, zmax, scale);
        }
    }

    let p0, p1, p2, p3;
    let vertex = []

    for (let i = 0; i < nu - 1; i++) {
        for (let j = 0; j < nv - 1; j++) {
            p0 = pts[i][j];
            p1 = pts[i + 1][j];
            p2 = pts[i + 1][j + 1];
            p3 = pts[i][j + 1];
            vertex.push([
                p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2],
                p2[0], p2[1], p2[2], p3[0], p3[1], p3[2], p0[0], p0[1], p0[2]].flat());
        }
    }
    return new Float32Array(vertex.flat())
};

const ParametricNormalData = (f, umin, umax, vmin, vmax, nu, nv,
                              xmin, xmax, zmin, zmax, scale = 1, scaley = 0, center = [0, 0, 0]) => {
    const du = (umax - umin) / (nu - 1);
    const dv = (vmax - vmin) / (nv - 1);
    let pts = [];
    let u, v;
    let pt;
    let ymin1 = 0, ymax1 = 0;

    for (let i = 0; i < nu; i++) {
        u = umin + i * du;
        let pt1 = [];
        for (let j = 0; j < nv; j++) {
            v = vmin + j * dv;
            pt = f(u, v, center);
            let n = NormalToPoint(u,v)
            ymin1 = (pt[1] < ymin1) ? pt[1] : ymin1;
            ymax1 = (pt[1] > ymax1) ? pt[1] : ymax1;
            pt1.push(n);

        }
        pts.push(pt1);

    }

    const ymin = ymin1 - scaley * (ymax1 - ymin1);
    const ymax = ymax1 + scaley * (ymax1 - ymin1);

    for (let i = 0; i < nu; i++) {
        for (let j = 0; j < nv; j++) {
            pts[i][j] = NormalizePoint(pts[i][j], xmin, xmax, ymin, ymax, zmin, zmax, scale);
        }
    }

    let p0, p1, p2, p3;
    let vertex = []

    for (let i = 0; i < nu - 1; i++) {
        for (let j = 0; j < nv - 1; j++) {
            p0 = pts[i][j];
            p1 = pts[i + 1][j];
            p2 = pts[i + 1][j + 1];
            p3 = pts[i][j + 1];
            vertex.push([
                p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2],
                p2[0], p2[1], p2[2], p3[0], p3[1], p3[2], p0[0], p0[1], p0[2]].flat());
        }
    }
    return new Float32Array(vertex.flat())
};

const NormalizePoint = (pt, xmin, xmax, ymin, ymax, zmin, zmax, scale = 1) => {
    pt[0] = scale * (-1 + 2 * (pt[0] - xmin) / (xmax - xmin));
    pt[1] = scale * (-1 + 2 * (pt[1] - ymin) / (ymax - ymin));
    pt[2] = scale * (-1 + 2 * (pt[2] - zmin) / (zmax - zmin));
    return pt;
};
let e = 0.0001;
let NormalToPoint = (u, v) => {
    let nu = KleinBottle(u, v),
        uN = KleinBottle(u + e, v),
        vN = KleinBottle(u, v + e),
        dU = [],
        dV = [];
    for (let i = 0; i < 3; i++) {
        dU.push((nu[i] - uN[i]) / e)
        dV.push((nu[i] - vN[i]) / e)
    }
    let n = m4.normalize(m4.cross(dU, dV))
    return n;
}

function deg2rad(angle) {
    return angle * Math.PI / 180;
}

// Constructor
function Model(name) {
    this.name = name;
    this.iVertexBuffer = gl.createBuffer();
    this.iNormalBuffer = gl.createBuffer();
    this.count = 0;

    this.BufferData = function (vertices) {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STREAM_DRAW);

        this.count = vertices.length / 3;
    }
    this.NormalData = function (normals) {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iNormalBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STREAM_DRAW);

    }

    this.Draw = function () {

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.vertexAttribPointer(shProgram.iAttribVertex, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribVertex);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iNormalBuffer);
        gl.vertexAttribPointer(shProgram.iAttribNormal, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribNormal);

        gl.drawArrays(gl.TRIANGLES, 0, this.count);
    }
}


// Constructor
function ShaderProgram(name, program) {

    this.name = name;
    this.prog = program;

    // Location of the attribute variable in the shader program.
    this.iAttribVertex = -1;
    // Location of the uniform specifying a color for the primitive.
    this.iColor = -1;
    // Location of the uniform matrix representing the combined transformation.
    this.iModelViewProjectionMatrix = -1;

    this.Use = function () {
        gl.useProgram(this.prog);
    }
}


/* Draws a colored cube, along with a set of coordinate axes.
 * (Note that the use of the above drawPrimitive function is not an efficient
 * way to draw with WebGL.  Here, the geometry is so simple that it doesn't matter.)
 */
function draw() {
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    /* Set the values of the projection transformation */
    // let projection = m4.perspective(Math.PI / 8, 1, 8, 12);
    let projection = m4.orthographic(-3, 3, -3, 3, -3, 3);

    /* Get the view matrix from the SimpleRotator object.*/
    let modelView = spaceball.getViewMatrix();

    let rotateToPointZero = m4.axisRotation([0.707, 0.707, 0], 0.7);
    let translateToPointZero = m4.translation(0, 0, -1);

    let matAccum0 = m4.multiply(rotateToPointZero, modelView);
    let matAccum1 = m4.multiply(translateToPointZero, matAccum0);

    /* Multiply the projection matrix times the modelview matrix to give the
       combined transformation matrix, and send that to the shader program. */
    let modelViewProjection = m4.multiply(projection, matAccum1);

    gl.uniformMatrix4fv(shProgram.iModelViewProjectionMatrix, false, modelViewProjection);
    let p = [
        document.getElementById('px').value,
        document.getElementById('py').value,
        document.getElementById('pz').value,
    ]
    let d = [
        document.getElementById('dx').value,
        document.getElementById('dy').value,
        document.getElementById('dz').value,
    ]
    gl.uniform3fv(shProgram.iPos, p);
    gl.uniform3fv(shProgram.iDir, d);
    gl.uniform1f(shProgram.iRange, document.getElementById('r').value);
    gl.uniform1f(shProgram.iFocus, document.getElementById('f').value);

    /* Draw the six faces of a cube, with different colors. */
    gl.uniform4fv(shProgram.iColor, [1, 1, 0, 1]);

    surface.Draw();
}

function CreateSurfaceData() {
    let vertexList = [];

    for (let i = 0; i < 360; i += 5) {
        vertexList.push(Math.sin(deg2rad(i)), 1, Math.cos(deg2rad(i)));
        vertexList.push(Math.sin(deg2rad(i)), 0, Math.cos(deg2rad(i)));
    }

    return vertexList;
}


/* Initialize the WebGL context. Called from init() */
function initGL() {
    let prog = createProgram(gl, vertexShaderSource, fragmentShaderSource);

    shProgram = new ShaderProgram('Basic', prog);
    shProgram.Use();

    shProgram.iAttribVertex = gl.getAttribLocation(prog, "vertex");
    shProgram.iAttribNormal = gl.getAttribLocation(prog, "normal");
    shProgram.iModelViewProjectionMatrix = gl.getUniformLocation(prog, "ModelViewProjectionMatrix");
    shProgram.iColor = gl.getUniformLocation(prog, "color");
    shProgram.iPos = gl.getUniformLocation(prog, "pos");
    shProgram.iDir = gl.getUniformLocation(prog, "dir");
    shProgram.iRange = gl.getUniformLocation(prog, "range");
    shProgram.iFocus = gl.getUniformLocation(prog, "focus");

    surface = new Model('Surface');
    surface.BufferData(ParametricSurfaceData(KleinBottle, 0, Math.PI, 0, 2 * Math.PI, 50, 15, -2, 2, -2, 2, 2, 0, [0, 0, 0]));
    surface.NormalData(ParametricNormalData(KleinBottle, 0, Math.PI, 0, 2 * Math.PI, 50, 15, -2, 2, -2, 2, 2, 0, [0, 0, 0]));

    gl.enable(gl.DEPTH_TEST);
}


/* Creates a program for use in the WebGL context gl, and returns the
 * identifier for that program.  If an error occurs while compiling or
 * linking the program, an exception of type Error is thrown.  The error
 * string contains the compilation or linking error.  If no error occurs,
 * the program identifier is the return value of the function.
 * The second and third parameters are strings that contain the
 * source code for the vertex shader and for the fragment shader.
 */
function createProgram(gl, vShader, fShader) {
    let vsh = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vsh, vShader);
    gl.compileShader(vsh);
    if (!gl.getShaderParameter(vsh, gl.COMPILE_STATUS)) {
        throw new Error("Error in vertex shader:  " + gl.getShaderInfoLog(vsh));
    }
    let fsh = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fsh, fShader);
    gl.compileShader(fsh);
    if (!gl.getShaderParameter(fsh, gl.COMPILE_STATUS)) {
        throw new Error("Error in fragment shader:  " + gl.getShaderInfoLog(fsh));
    }
    let prog = gl.createProgram();
    gl.attachShader(prog, vsh);
    gl.attachShader(prog, fsh);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error("Link error in program:  " + gl.getProgramInfoLog(prog));
    }
    return prog;
}


/**
 * initialization function that will be called when the page has loaded
 */
function init() {
    let canvas;
    try {
        canvas = document.getElementById("webglcanvas");
        gl = canvas.getContext("webgl");
        if (!gl) {
            throw "Browser does not support WebGL";
        }
    }
    catch (e) {
        document.getElementById("canvas-holder").innerHTML =
            "<p>Sorry, could not get a WebGL graphics context.</p>";
        return;
    }
    try {
        initGL();  // initialize the WebGL graphics context
    }
    catch (e) {
        document.getElementById("canvas-holder").innerHTML =
            "<p>Sorry, could not initialize the WebGL graphics context: " + e + "</p>";
        return;
    }

    spaceball = new TrackballRotator(canvas, draw, 0);

    draw();
}
