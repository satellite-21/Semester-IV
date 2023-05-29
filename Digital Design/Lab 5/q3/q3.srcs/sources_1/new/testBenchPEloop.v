`timescale 1ns / 1ps
module testBench;

reg [3:0] d;

wire [1:0] x;
wire v;

priorityEncoder pe(d,x,v);

initial
 begin
  d = 4'b0000;
  #10 d = 4'b1000;
  #10 d = 4'b0100;
  #10 d = 4'b0010;
  #10 d = 4'b0001;
 end 
initial #50 $finish;

endmodule

