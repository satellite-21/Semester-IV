`timescale 1ns / 1ps

module full_adder(A,B,C,s_ans,c_ans);
input A,B,C;
output s_ans,c_ans;

half_adder H1(B,C,s1,c1);
half_adder H2(A,s1,s2,c2);

assign s_ans = s2;
assign c_ans = c1|c2;

endmodule
