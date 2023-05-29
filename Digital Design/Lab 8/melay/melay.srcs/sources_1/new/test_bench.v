`timescale 1ns / 1ps
module test_melay_2s_complement;
reg rst, clk;
wire out, inp;
wire [1:0] ps, ns;
MSL_using_LFSR mul(inp, rst, clk);
melay_2s_complement m2s(inp, clk, rst, out, ps, ns);
initial
begin
rst = 1;
clk = 0;
#1 rst = 0;
end
always #2 clk = clk^1;
endmodule