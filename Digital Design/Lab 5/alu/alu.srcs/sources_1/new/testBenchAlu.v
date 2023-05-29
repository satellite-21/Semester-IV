`timescale 1ns / 1ps
module testBench;
 reg [31:0] a,b;
 reg [2:0] op;
 wire [31:0] result;

 alu abc (a,b,op,result);

 initial 
  begin
  a = 31'b00000000000000000000000000001010;
  b = 31'b00000000000000000000000000000110;
  op = 0;
  #10 op = 1;
  #10 op = 2;
  #10 op = 3;
  #10 op = 4;
  #10 op = 5;
  #10 op = 6;
  #10 op = 7;
  end
 initial #100 $finish;

endmodule


