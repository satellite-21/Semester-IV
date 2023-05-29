`timescale 1ns / 1ps


module mux(i1,i2,i3,i4,op,control);
input i1,i2,i3,i4;
input [1:0] control;
output reg op;

always @*
    begin
    case(control)
    2'b00: op = i1;
    2'b01: op = i2;
    2'b10: op = i3;
    2'b11: op = i4;
    default : op=1'bz;
    endcase
    end
    

endmodule
