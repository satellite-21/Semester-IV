`timescale 1ns / 1ps


module bcd_to_sevenseg(bcd,display);

    input[0:2]bcd;
    output[0:6]display;
    
    reg [0:6]display;
    
        always@(bcd)
            case(bcd)
                
                0 : display = 7'b1111110;
                1 : display = 7'b0110000;
                2 : display = 7'b1101101;
                3 : display = 7'b1111001;
                4 : display = 7'b0110011;
                5 : display = 7'b1011011;
                6 : display = 7'b1011111;
                7 : display = 7'b1110000;
                8 : display = 7'b1111111;
                9 : display = 7'b1111011;
                
            default : display = 7'b0000000;
            endcase
endmodule
