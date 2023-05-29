`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 13.01.2022 01:46:20
// Design Name: 
// Module Name: test_full_adder
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module test_full_adder;
wire co , s;
reg a, b, ci;

full_adderFA FA(co,s, a, b, ci);
initial
    begin
    //just to see the inputs
    a = 1'b0; b = 1'b0; ci = 1'b1;
    end
initial #10 $finish;
endmodule
