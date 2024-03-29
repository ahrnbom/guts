inputs:  x3D, y3D, P11_1, P11_2, P11_3, P11_4, P12_1, P12_2, P12_3, P12_4, P13_1, P13_2, P13_3, P13_4, P21_1, P21_2, P21_3, P21_4, P22_1, P22_2, P22_3, P22_4, P23_1, P23_2, P23_3, P23_4, x11, x12, x13, x21, x22, x23, G1_1, G1_2, G1_3, G1_4, G1_5, G1_6, G1_7, G1_8, G2_1, G2_2, G2_3, G2_4, G2_5, G2_6, G2_7, G2_8, G3_1, G3_2, G3_3, G3_4, G3_5, G3_6, G3_7, G3_8, alpha
Matrix output: 4  2
t2 = P11_1*x3D;
t3 = P12_1*x3D;
t4 = P13_1*x3D;
t5 = P21_1*x3D;
t6 = P22_1*x3D;
t7 = P23_1*x3D;
t8 = P11_2*y3D;
t9 = P12_2*y3D;
t10 = P13_2*y3D;
t11 = P21_2*y3D;
t12 = P22_2*y3D;
t13 = P23_2*y3D;
t14 = G1_1*2.0;
t15 = G1_2*2.0;
t16 = G1_3*2.0;
t17 = G1_4*2.0;
t18 = G1_5*2.0;
t19 = G1_6*2.0;
t20 = G1_7*2.0;
t21 = G1_8*2.0;
t22 = G2_1*2.0;
t23 = G2_2*2.0;
t24 = G2_3*2.0;
t25 = G2_4*2.0;
t26 = G2_5*2.0;
t27 = G2_6*2.0;
t28 = G2_7*2.0;
t29 = G2_8*2.0;
t30 = x3D*2.0;
t31 = y3D*2.0;
t32 = -x3D;
t34 = -y3D;
t33 = -t30;
t35 = -t31;
t36 = G1_1+t32;
t37 = G1_2+t32;
t38 = G1_3+t32;
t39 = G1_4+t32;
t40 = G1_5+t32;
t41 = G1_6+t32;
t42 = G1_7+t32;
t43 = G1_8+t32;
t44 = G2_1+t34;
t45 = G2_2+t34;
t46 = G2_3+t34;
t47 = G2_4+t34;
t48 = G2_5+t34;
t49 = G2_6+t34;
t50 = G2_7+t34;
t51 = G2_8+t34;
t52 = t14+t33;
t53 = t15+t33;
t54 = t16+t33;
t55 = t17+t33;
t56 = t18+t33;
t57 = t19+t33;
t58 = t20+t33;
t59 = t21+t33;
t60 = t36*t36;
t61 = t37*t37;
t62 = t38*t38;
t63 = t39*t39;
t64 = t40*t40;
t65 = t41*t41;
t66 = t42*t42;
t67 = t43*t43;
t68 = t22+t35;
t69 = t23+t35;
t70 = t24+t35;
t71 = t25+t35;
t72 = t26+t35;
t73 = t27+t35;
t74 = t28+t35;
t75 = t29+t35;
t76 = t44*t44;
t77 = t45*t45;
t78 = t46*t46;
t79 = t47*t47;
t80 = t48*t48;
t81 = t49*t49;
t82 = t50*t50;
t83 = t51*t51;
t84 = t60+t76;
t85 = t61+t77;
t86 = t62+t78;
t87 = t63+t79;
t88 = t64+t80;
t89 = t65+t81;
t90 = t66+t82;
t91 = t67+t83;
t92 = sqrt(t84);
t93 = sqrt(t85);
t94 = sqrt(t86);
t95 = sqrt(t87);
t96 = sqrt(t88);
t97 = sqrt(t89);
t98 = sqrt(t90);
t99 = sqrt(t91);
t100 = 1.0/t92;
t101 = 1.0/t93;
t102 = 1.0/t94;
t103 = 1.0/t95;
t104 = 1.0/t96;
t105 = 1.0/t97;
t106 = 1.0/t98;
t107 = 1.0/t99;
t108 = alpha*t92;
t109 = alpha*t93;
t110 = alpha*t94;
t111 = alpha*t95;
t112 = alpha*t96;
t113 = alpha*t97;
t114 = alpha*t98;
t115 = alpha*t99;
t116 = -t108;
t117 = -t109;
t118 = -t110;
t119 = -t111;
t120 = -t112;
t121 = -t113;
t122 = -t114;
t123 = -t115;
t124 = exp(t116);
t125 = exp(t117);
t126 = exp(t118);
t127 = exp(t119);
t128 = exp(t120);
t129 = exp(t121);
t130 = exp(t122);
t131 = exp(t123);
t132 = G3_1*t124;
t133 = G3_2*t125;
t134 = G3_3*t126;
t135 = G3_4*t127;
t136 = G3_5*t128;
t137 = G3_6*t129;
t138 = G3_7*t130;
t139 = G3_8*t131;
t140 = (alpha*t52*t100*t124)/2.0;
t141 = (alpha*t53*t101*t125)/2.0;
t142 = (alpha*t54*t102*t126)/2.0;
t143 = (alpha*t55*t103*t127)/2.0;
t144 = (alpha*t56*t104*t128)/2.0;
t145 = (alpha*t57*t105*t129)/2.0;
t146 = (alpha*t58*t106*t130)/2.0;
t147 = (alpha*t59*t107*t131)/2.0;
t148 = (alpha*t68*t100*t124)/2.0;
t149 = (alpha*t69*t101*t125)/2.0;
t150 = (alpha*t70*t102*t126)/2.0;
t151 = (alpha*t71*t103*t127)/2.0;
t152 = (alpha*t72*t104*t128)/2.0;
t153 = (alpha*t73*t105*t129)/2.0;
t154 = (alpha*t74*t106*t130)/2.0;
t155 = (alpha*t75*t107*t131)/2.0;
t172 = t124+t125+t126+t127+t128+t129+t130+t131;
t156 = (alpha*t52*t100*t132)/2.0;
t157 = (alpha*t53*t101*t133)/2.0;
t158 = (alpha*t54*t102*t134)/2.0;
t159 = (alpha*t55*t103*t135)/2.0;
t160 = (alpha*t56*t104*t136)/2.0;
t161 = (alpha*t57*t105*t137)/2.0;
t162 = (alpha*t58*t106*t138)/2.0;
t163 = (alpha*t59*t107*t139)/2.0;
t164 = (alpha*t68*t100*t132)/2.0;
t165 = (alpha*t69*t101*t133)/2.0;
t166 = (alpha*t70*t102*t134)/2.0;
t167 = (alpha*t71*t103*t135)/2.0;
t168 = (alpha*t72*t104*t136)/2.0;
t169 = (alpha*t73*t105*t137)/2.0;
t170 = (alpha*t74*t106*t138)/2.0;
t171 = (alpha*t75*t107*t139)/2.0;
t173 = 1.0/t172;
t175 = t132+t133+t134+t135+t136+t137+t138+t139;
t192 = t140+t141+t142+t143+t144+t145+t146+t147;
t193 = t148+t149+t150+t151+t152+t153+t154+t155;
t174 = t173*t173;
t176 = P11_3*t173*t175;
t177 = P12_3*t173*t175;
t178 = P13_3*t173*t175;
t179 = P21_3*t173*t175;
t180 = P22_3*t173*t175;
t181 = P23_3*t173*t175;
t194 = t156+t157+t158+t159+t160+t161+t162+t163;
t195 = t164+t165+t166+t167+t168+t169+t170+t171;
t182 = P11_4+t2+t8+t176;
t183 = P12_4+t3+t9+t177;
t184 = P13_4+t4+t10+t178;
t185 = P21_4+t5+t11+t179;
t186 = P22_4+t6+t12+t180;
t187 = P23_4+t7+t13+t181;
t196 = P13_3*t173*t194;
t197 = P23_3*t173*t194;
t198 = P13_3*t173*t195;
t199 = P23_3*t173*t195;
t200 = P13_3*t174*t175*t192;
t201 = P23_3*t174*t175*t192;
t202 = P13_3*t174*t175*t193;
t203 = P23_3*t174*t175*t193;
t188 = 1.0/t184;
t190 = 1.0/t187;
t204 = -t200;
t205 = -t201;
t206 = -t202;
t207 = -t203;
t189 = t188*t188;
t191 = t190*t190;
t208 = P13_1+t196+t204;
t209 = P23_1+t197+t205;
t210 = P13_2+t198+t206;
t211 = P23_2+t199+t207;
A0[0][0] = t188*(P11_1+P11_3*t173*t194-P11_3*t174*t175*t192)-t182*t189*t208;
A0[0][1] = t188*(P11_2+P11_3*t173*t195-P11_3*t174*t175*t193)-t182*t189*t210;
A0[1][0] = t188*(P12_1+P12_3*t173*t194-P12_3*t174*t175*t192)-t183*t189*t208;
A0[1][1] = t188*(P12_2+P12_3*t173*t195-P12_3*t174*t175*t193)-t183*t189*t210;
A0[2][0] = t190*(P21_1+P21_3*t173*t194-P21_3*t174*t175*t192)-t185*t191*t209;
A0[2][1] = t190*(P21_2+P21_3*t173*t195-P21_3*t174*t175*t193)-t185*t191*t211;
A0[3][0] = t190*(P22_1+P22_3*t173*t194-P22_3*t174*t175*t192)-t186*t191*t209;
A0[3][1] = t190*(P22_2+P22_3*t173*t195-P22_3*t174*t175*t193)-t186*t191*t211;
