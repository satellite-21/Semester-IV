`timescale 1ns / 1ps
module test_moore_2s_complement;
reg rst, clk;
wire out, inp;
wire [1:0] state;
MSL_using_LSFR mul(inp, rst, clk);
moore_2s_complement m2s(inp, clk, rst, out, state);
initial begin
rst = 1;
clk = 0;
#1 rst = 0;
end
always #2 clk = clk^1;
endmodule