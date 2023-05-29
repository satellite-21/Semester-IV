`timescale 1ns / 1ps

module carrySelectAdder32bit(a,b,s,cout);
input [31:0]a;
input [31:0]b;
output [31:0]s;
output cout;
assign c0=1'b0;
carrySelectAdder csa1(s[3:0],c1,a[3:0],b[3:0],c0);
carrySelectAdder csa2(s[7:4],c2,a[7:4],b[7:4],c1);
carrySelectAdder csa3(s[11:8],c3,a[11:8],b[11:8],c2);
carrySelectAdder csa4(s[15:12],c4,a[15:12],b[15:12],c3);
carrySelectAdder csa5(s[19:16],c5,a[19:16],b[19:16],c4);
carrySelectAdder csa6(s[23:20],c6,a[23:20],b[23:20],c5);
carrySelectAdder csa7(s[27:24],c7,a[27:24],b[27:24],c6);
carrySelectAdder csa8(s[31:28],cout,a[31:28],b[31:28],c7);
endmodule

module carrySelectAdder(output [3:0]sum,output cout,input [3:0]a,input[3:0]b,input cin);
full_adder a1(a[0],b[0],cin,sum[0],c1);
full_adder a2(a[1],b[1],c1,sum[1],c2);
full_adder a3(a[2],b[2],1'b0,s3,c3);
full_adder a4(a[3],b[3],c3,s4,c4);
full_adder a5(a[2],b[2],1'b1,s5,c5);
full_adder a6(a[3],b[3],c5,s6,c6);
assign sum[2]= (!c2?s3:s5);
assign sum[3]= (!c2?s4:s6);
assign cout= (!c2?c4:c6);
endmodule

module full_adder(a,b,cin,sum,cout);
  input a,b,cin;
  output sum,cout;

  half_adder h1(a,b,x,y);
  half_adder h2(x,cin,sum,z);
  or (cout,y,z);
endmodule 

module half_adder (bit1,bit2,sum,carry);
  input  bit1;
  input  bit2;
  output sum;
  output carry;
 
  assign sum   = bit1 ^ bit2;  
  assign carry = bit1 & bit2; 
 
endmodule

