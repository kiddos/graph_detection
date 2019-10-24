import { Component, OnInit, ElementRef, ViewChild } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';

import { Point } from '../point';
import { BBox } from '../bbox';
import { Delta } from '../delta';
import { Label } from '../label';

@Component({
  selector: 'app-label',
  templateUrl: './label.component.html',
  styleUrls: ['./label.component.css']
})
export class LabelComponent implements OnInit {
  @ViewChild('canvas', { static: true })
  canvas: ElementRef<HTMLCanvasElement>;

  private context: CanvasRenderingContext2D;

  private mode: string;
  private labeling: boolean;
  private editing: boolean;
  private currentBBox?: BBox;
  private currentImage?: CanvasImageSource;
  private currentHighlight?: Point;
  private delta?: Delta;
  private labels: Array<Label>;
  private currentLabel?: Label;

  constructor(public dialog: MatDialog) {
    this.mode = 'Add';
    this.labeling = false;
    this.labels = [];
  }

  ngOnInit() {
    this.context = this.canvas.nativeElement.getContext('2d');
  }

  switchMode(event) : void {
    this.mode = event.value;
    this.redraw();
  }

  clearLabels() : void {
    if (this.currentLabel) {
      this.currentLabel.clearBBoxes();

      this.redraw();
    }
  }

  getPoint(event: MouseEvent) : Point {
    let rect = this.canvas.nativeElement.getBoundingClientRect();
    let x = event.clientX - rect.left;
    let y = event.clientY - rect.top;
    x = x * this.canvas.nativeElement.width / rect.width;
    y = y * this.canvas.nativeElement.height / rect.height;
    return new Point(x, y);
  }

  handleMouseDown(event: MouseEvent) : void {
    if (this.mode === 'Add') {
      this.labeling = true;
      let p1 = this.getPoint(event)
      let p4 = this.getPoint(event)
      this.currentBBox = new BBox(p1, p4);
      this.redraw();
    }

    if (this.mode === 'Edit') {
      let current = this.getPoint(event);
      if (this.currentLabel) {
        let p = this.currentLabel.findClosest(current);
        if (p) {
          this.delta = new Delta(current, current);
          this.editing = true;
        }
      }
    }
  }

  handleMouseMove(event: MouseEvent) : void {
    let current = this.getPoint(event)
    if (this.labeling) {
      this.currentBBox.p3 = current;
      this.currentBBox.p2.x = current.x;
      this.currentBBox.p4.y = current.y;

      this.redraw();
    }

    if (this.mode === 'Edit') {
      let last = this.currentHighlight;

      if (this.currentLabel) {
        this.currentHighlight = this.currentLabel.findClosest(current);
      }

      if (this.delta && this.editing) {
        this.delta.end = current;

        if (this.currentHighlight) {
          this.currentHighlight.x += this.delta.getDx();
          this.currentHighlight.y += this.delta.getDy();
        }
        this.delta.start = current;
      }

      if (last !== this.currentHighlight || this.editing) {
        this.redraw();
      }
    }
  }

  handleMouseUp(event: MouseEvent) : void {
    if (this.labeling) {
      this.labeling = false;
      if (this.currentLabel) {
        this.currentLabel.addBBox(this.currentBBox);
      }
      this.redraw();
    }

    if (this.editing) {
      this.editing = false;
      this.redraw();
    }
  }

  redraw() : void {
    this.context.clearRect(0, 0, this.canvas.nativeElement.width,
      this.canvas.nativeElement.height);

    if (this.currentImage) {
      this.context.drawImage(this.currentImage, 0, 0,
        this.canvas.nativeElement.width, this.canvas.nativeElement.height);
    }

    if (this.labeling) {
      this.currentBBox.draw(this.context, this.mode);
    }

    if (this.currentLabel) {
      for (let i = 0; i < this.currentLabel.bboxes.length; ++i) {
        this.currentLabel.bboxes[i].draw(this.context, this.mode);
      }
    }

    if (this.currentHighlight) {
      this.currentHighlight.highlight(this.context, this.mode);
    }
  }

  createNewLabel(name: string) : void {
    this.currentLabel = new Label();
    this.currentLabel.imageName = name;
    this.labels.push(this.currentLabel);
  }

  handleFile(files: FileList) : void {
    if (files.length > 0) {
      let f = files[0];
      this.currentImage = new Image();
      this.currentImage.src = URL.createObjectURL(f);

      this.currentImage.onload = () => {
        this.context.drawImage(this.currentImage, 0, 0,
          this.canvas.nativeElement.width, this.canvas.nativeElement.height);
        this.redraw();
      };

      if (this.labels.length > 0) {
        let found = false;
        for (let i = 0; i < this.labels.length; ++i) {
          if (this.labels[i].imageName === f.name) {
            found = true;
            this.currentLabel = this.labels[i];
            break;
          }
        }

        if (!found) {
          this.createNewLabel(f.name);
        }
      } else {
        this.createNewLabel(f.name);
      }
    }
  }

  saveLabels() : void {
    if (this.currentLabel) {
      let jsonString = JSON.stringify(this.labels);

      let tmp = document.createElement('a');
      tmp.setAttribute('href', 'data:text/json;charset=UTF-8,' + encodeURIComponent(jsonString));
      tmp.setAttribute('download', 'labels.json');
      tmp.style.display = 'none';
      document.body.appendChild(tmp);
      tmp.click();
      document.body.removeChild(tmp);
    }
  }

  openHelp() : void {
    this.dialog.open(HelpDialog);
  }

  loadLabels(files: FileList) : void {
    if (files.length > 0) {
      let f = files[0];

      let fr = new FileReader();
      fr.onload = (e) => {
        let content = fr.result.toString();
        let label = JSON.parse(content);

        for (let i = 0; i < label.length; ++i) {
          let found = false;
          for (let j = 0; j < this.labels.length; ++j) {
            if (this.labels[j].imageName === label[i].imageName) {
              let newLabel = Label.fromJSONObject(label[i]);
              this.labels[j].bboxes = this.labels[j].bboxes.concat(newLabel.bboxes);
              found = true;
            }
          }

          if (!found) {
            this.labels.push(Label.fromJSONObject(label[i]));
          }
        }
        this.redraw();
      }

      fr.readAsText(f);
    }
  }
}

@Component({
  selector: 'help-dialog',
  templateUrl: './help.html',
})
export class HelpDialog {
  constructor() {}
}
