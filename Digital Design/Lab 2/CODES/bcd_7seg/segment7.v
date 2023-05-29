`timescale 1ns / 1ps
module segment7(
     bcd,
     seg
    );
     
     //Declare inputs,outputs and internal variables.
     input [0:3]bcd;
     output [0:6]seg;
     reg [0:6]seg;

//always block for converting bcd digit into 7 segment format
    always @(bcd)
        case (bcd) //case statement
            0:seg=7'b1111110;
            1:seg=7'b0110000;
            2:seg=7'b1101101;
            3:seg=7'b1111001;
            4:seg=7'b0111001;
            5:seg=7'b1011011;
            6:seg=7'b1011111;
            7:seg=7'b1110000;
            8:seg=7'b1111111;
            9:seg=7'b1111011;
            10:seg=7'b1110111;
            11:seg=7'b0011111;
            12:seg=7'b1001110;
            13:seg=7'b0111101;
            14:seg=7'b1001111;
            15:seg=7'b1000111;
            default : seg = 7'b0000000; 
        endcase
    
endmodule

