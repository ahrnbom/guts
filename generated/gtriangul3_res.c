#include <math.h>

void generated_function(double* A0, const double x3D, const double y3D, const double P11_1, const double P11_2, const double P11_3, const double P11_4, const double P12_1, const double P12_2, const double P12_3, const double P12_4, const double P13_1, const double P13_2, const double P13_3, const double P13_4, const double P21_1, const double P21_2, const double P21_3, const double P21_4, const double P22_1, const double P22_2, const double P22_3, const double P22_4, const double P23_1, const double P23_2, const double P23_3, const double P23_4, const double P31_1, const double P31_2, const double P31_3, const double P31_4, const double P32_1, const double P32_2, const double P32_3, const double P32_4, const double P33_1, const double P33_2, const double P33_3, const double P33_4, const double x11, const double x12, const double x13, const double x21, const double x22, const double x23, const double x31, const double x32, const double x33, const double G1_1, const double G1_2, const double G1_3, const double G1_4, const double G1_5, const double G1_6, const double G1_7, const double G1_8, const double G2_1, const double G2_2, const double G2_3, const double G2_4, const double G2_5, const double G2_6, const double G2_7, const double G2_8, const double G3_1, const double G3_2, const double G3_3, const double G3_4, const double G3_5, const double G3_6, const double G3_7, const double G3_8, const double alpha) {
    const double t2 = P13_1*x3D;
    const double t3 = P23_1*x3D;
    const double t4 = P33_1*x3D;
    const double t5 = P13_2*y3D;
    const double t6 = P23_2*y3D;
    const double t7 = P33_2*y3D;
    const double t8 = -x3D;
    const double t9 = 1.0/x13;
    const double t10 = 1.0/x23;
    const double t11 = 1.0/x33;
    const double t12 = -y3D;
    const double t13 = G1_1+t8;
    const double t14 = G1_2+t8;
    const double t15 = G1_3+t8;
    const double t16 = G1_4+t8;
    const double t17 = G1_5+t8;
    const double t18 = G1_6+t8;
    const double t19 = G1_7+t8;
    const double t20 = G1_8+t8;
    const double t21 = G2_1+t12;
    const double t22 = G2_2+t12;
    const double t23 = G2_3+t12;
    const double t24 = G2_4+t12;
    const double t25 = G2_5+t12;
    const double t26 = G2_6+t12;
    const double t27 = G2_7+t12;
    const double t28 = G2_8+t12;
    const double t29 = t13*t13;
    const double t30 = t14*t14;
    const double t31 = t15*t15;
    const double t32 = t16*t16;
    const double t33 = t17*t17;
    const double t34 = t18*t18;
    const double t35 = t19*t19;
    const double t36 = t20*t20;
    const double t37 = t21*t21;
    const double t38 = t22*t22;
    const double t39 = t23*t23;
    const double t40 = t24*t24;
    const double t41 = t25*t25;
    const double t42 = t26*t26;
    const double t43 = t27*t27;
    const double t44 = t28*t28;
    const double t45 = t29+t37;
    const double t46 = t30+t38;
    const double t47 = t31+t39;
    const double t48 = t32+t40;
    const double t49 = t33+t41;
    const double t50 = t34+t42;
    const double t51 = t35+t43;
    const double t52 = t36+t44;
    const double t53 = sqrt(t45);
    const double t54 = sqrt(t46);
    const double t55 = sqrt(t47);
    const double t56 = sqrt(t48);
    const double t57 = sqrt(t49);
    const double t58 = sqrt(t50);
    const double t59 = sqrt(t51);
    const double t60 = sqrt(t52);
    const double t61 = alpha*t53;
    const double t62 = alpha*t54;
    const double t63 = alpha*t55;
    const double t64 = alpha*t56;
    const double t65 = alpha*t57;
    const double t66 = alpha*t58;
    const double t67 = alpha*t59;
    const double t68 = alpha*t60;
    const double t69 = -t61;
    const double t70 = -t62;
    const double t71 = -t63;
    const double t72 = -t64;
    const double t73 = -t65;
    const double t74 = -t66;
    const double t75 = -t67;
    const double t76 = -t68;
    const double t77 = exp(t69);
    const double t78 = exp(t70);
    const double t79 = exp(t71);
    const double t80 = exp(t72);
    const double t81 = exp(t73);
    const double t82 = exp(t74);
    const double t83 = exp(t75);
    const double t84 = exp(t76);
    const double t85 = G3_1*t77;
    const double t86 = G3_2*t78;
    const double t87 = G3_3*t79;
    const double t88 = G3_4*t80;
    const double t89 = G3_5*t81;
    const double t90 = G3_6*t82;
    const double t91 = G3_7*t83;
    const double t92 = G3_8*t84;
    const double t93 = t77+t78+t79+t80+t81+t82+t83+t84;
    const double t94 = 1.0/t93;
    const double t95 = t85+t86+t87+t88+t89+t90+t91+t92;
    const double t96 = P13_3*t94*t95;
    const double t97 = P23_3*t94*t95;
    const double t98 = P33_3*t94*t95;
    const double t99 = P13_4+t2+t5+t96;
    const double t100 = P23_4+t3+t6+t97;
    const double t101 = P33_4+t4+t7+t98;
    const double t102 = 1.0/t99;
    const double t103 = 1.0/t100;
    const double t104 = 1.0/t101;
    A0[0] = -t9*x11+t102*(P11_4+P11_1*x3D+P11_2*y3D+P11_3*t94*t95);
    A0[1] = -t9*x12+t102*(P12_4+P12_1*x3D+P12_2*y3D+P12_3*t94*t95);
    A0[2] = -t10*x21+t103*(P21_4+P21_1*x3D+P21_2*y3D+P21_3*t94*t95);
    A0[3] = -t10*x22+t103*(P22_4+P22_1*x3D+P22_2*y3D+P22_3*t94*t95);
    A0[4] = -t11*x31+t104*(P31_4+P31_1*x3D+P31_2*y3D+P31_3*t94*t95);
    A0[5] = -t11*x32+t104*(P32_4+P32_1*x3D+P32_2*y3D+P32_3*t94*t95);
}

void main(double* out, double* data) {
    generated_function(out, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31], data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39], data[40], data[41], data[42], data[43], data[44], data[45], data[46], data[47], data[48], data[49], data[50], data[51], data[52], data[53], data[54], data[55], data[56], data[57], data[58], data[59], data[60], data[61], data[62], data[63], data[64], data[65], data[66], data[67], data[68], data[69], data[70], data[71]);
}
/* Output matrix should be of size 1 x 6 */