`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.02.2022 07:59:43
// Design Name: 
// Module Name: tb_bcdadder
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


module tb_bcdadder;

    // Inputs
    reg [3:0] a;
    reg [3:0] b;
    reg carry_in;

    // Outputs
    wire [3:0] sum;
    wire carry;

    // Instantiate the Unit Under Test (UUT)
    bcd_adder uut (
        .a(a), 
        .b(b), 
        .carry_in(carry_in), 
        .sum(sum), 
        .carry(carry)
    );

    initial begin
        // Apply Inputs
        a = 0;  b = 0;  carry_in = 0;   #100;
        a = 6;  b = 9;  carry_in = 0;   #100;
        a = 3;  b = 3;  carry_in = 1;   #100;
        a = 4;  b = 5;  carry_in = 0;   #100;
        a = 8;  b = 2;  carry_in = 0;   #100;
        a = 9;  b = 9;  carry_in = 1;   #100;
    end
      
endmodule