`timescale 1ns / 1ps
module dff_blocking(D, clk, Q1, Q2);

input D, clk;
output reg Q1, Q2;

always @(posedge clk)
begin 
    Q1 = D;
    Q2 = Q1;
end
endmodule


