`timescale 1ns / 1ps

module CLA_Adder(a,b,s,cout);
input [31:0]a;
input [31:0]b;
output [31:0]s;

output cout;

assign c0=1'b0;
cla cla1(s[3:0],c1,a[3:0],b[3:0],c0);
cla cla2(s[7:4],c2,a[7:4],b[7:4],c1);
cla cla3(s[11:8],c3,a[11:8],b[11:8],c2);
cla cla4(s[15:12],c4,a[15:12],b[15:12],c3);
cla cla5(s[19:16],c5,a[19:16],b[19:16],c4);
cla cla6(s[23:20],c6,a[23:20],b[23:20],c5);
cla cla7(s[27:24],c7,a[27:24],b[27:24],c6);
cla cla8(s[31:28],cout,a[31:28],b[31:28],c7);
endmodule

module cla(s,cout,a,b,cin);
input [3:0]a,b;
input cin;
output [3:0]s;
output cout;
wire [3:0]g;
wire [3:0]p;
half_adder ha1(a[0],b[0],p[0],g[0]);
half_adder ha2(a[1],b[1],p[1],g[1]);
half_adder ha3(a[2],b[2],p[2],g[2]);
half_adder ha4(a[3],b[3],p[3],g[3]);

assign c0=cin;
assign #5 c1=g[0] | (p[0]&&c0);
assign #5 c2=g[1] | (p[1]&&g[0]) | (p[1]&&p[0]&&c0);
assign #5 c3=g[2] | (p[2]&&g[1]) | (p[2]&&p[1]&&g[0]) | (p[2]&&p[1]&&p[0]&&c0);
assign #5 cout=g[3] | (p[3]&&g[2]) | (p[3]&&p[2]&&g[1]) | (p[3]&&p[2]&&p[1]&&g[0]) | (p[3]&&p[2]&&p[1]&&p[0]&&c0);

xor #10 (s[0],p[0],c0);
xor #10 (s[1],p[1],c1);
xor #10 (s[2],p[2],c2);
xor #10 (s[3],p[3],c3);
endmodule
module half_adder (bit1,bit2,sum,carry);
  input  bit1;
  input  bit2;
  output sum;
  output carry;
 
  xor(sum ,bit1,bit2); 
  and(carry,bit1,bit2); 
 
endmodule

