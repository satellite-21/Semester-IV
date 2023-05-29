`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 13.01.2022 02:59:35
// Design Name: 
// Module Name: test_fourbitadder
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
wire co , so, c1, s1, c2, s2, c3, s3;
reg a0, b0, ci, a1, b1, a2, b2, a3, b3;

full_adder FA1(co,so, a0, b0, ci);
initial
    begin
    //just to see the inputs
    a0 = 1'b1; b0 = 1'b0; ci = 1'b0;
    end
initial #10 $finish;


full_adder FA2(c1, s1 , a1, b1, co);
initial
    begin
    a1 = 1'b0; b1 = 1'b0;
    end
initial #10 $finish;

full_adder FA3(c2, s2 , a2, b2, c1);
initial
    begin
    a2 = 1'b0; b2 = 1'b1;
    end
initial #10 $finish;

full_adder FA4(c3, s3 , a3, b3, c2);
initial
    begin
    a3 = 1'b1; b3 = 1'b1;
    end
initial #10 $finish;

endmodule