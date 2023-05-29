`timescale 1ns / 1ps

module CarrySkipAdder(input [31:0]a,input [31:0]b,output [31:0]s,output cout);
assign c0 = 1'b0;

carrySkip csa1(s[3:0],c1,a[3:0],b[3:0],c0);
carrySkip csa2(s[7:4],c2,a[7:4],b[7:4],c1);
carrySkip csa3(s[11:8],c3,a[11:8],b[11:8],c2);
carrySkip csa4(s[15:12],c4,a[15:12],b[15:12],c3);
carrySkip csa5(s[19:16],c5,a[19:16],b[19:16],c4);
carrySkip csa6(s[23:20],c6,a[23:20],b[23:20],c5);
carrySkip csa7(s[27:24],c7,a[27:24],b[27:24],c6);
carrySkip csa8(s[31:28],cout,a[31:28],b[31:28],c7);
endmodule


module carrySkip(output[3:0] sum,output cout,input[3:0]a,input[3:0]b,input cin);

assign c0 = cin;

full_adder f1 (a[0],b[0],c0,sum[0],c1);
full_adder f2 (a[1],b[1],c1,sum[1],c2);
full_adder f3 (a[2],b[2],c2,sum[2],c3);
full_adder f4 (a[3],b[3],c3,sum[3],c4);

and(temp1,sum[0],sum[1]);
and(temp2,sum[2],sum[3]);

and(sel,temp1,temp2);
assign cout = (sel ? c0 : c4);
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

