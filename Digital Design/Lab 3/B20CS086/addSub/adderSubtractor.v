`timescale 1ns / 1ps

module adderSubtractor(a0,a1,a2,a3,b0,b1,b2,b3,m,s0,s1,s2,s3,v);
input a0,a1,a2,a3,b0,b1,b2,b3,m;
output s0,s1,s2,s3,v;

wire c0,c1,c2,c3,c4,a0,a1,a2,a3,b0,b1,b2,b3;

assign c0 = m;

full_add f1(a0,b0^m,c0,s0,c1);
full_add f2(a1,b1^m,c1,s1,c2);
full_add f3(a2,b2^m,c2,s2,c3);
full_add f4(a3,b3^m,c3,s3,c4);

assign v = c3^c4;

endmodule

module full_add(A,B,C,s_ans,c_ans);
input A,B,C;
output s_ans,c_ans;

half_adder H1(B,C,s1,c1);
half_adder H2(A,s1,s2,c2);

assign s_ans = s2;
assign c_ans = c1|c2;

endmodule
           
module half_adder(a,b,s,c);
input a,b;
output s,c;

assign s = a^b;
assign c = a&b;

endmodule




