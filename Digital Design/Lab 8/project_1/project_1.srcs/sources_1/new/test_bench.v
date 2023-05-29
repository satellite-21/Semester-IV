`timescale 1ns / 1ps
module test_MSL_using_LSFR;
parameter n = 8;
wire bit;
wire [n-1:0] bits;
reg clk, rst;
MSL_using_LSFR mul(bit, rst, clk, bits);
initial
begin
clk = 0;
rst = 1;
#1 rst = 0;
end

always #2 clk = clk^1;
endmodule


